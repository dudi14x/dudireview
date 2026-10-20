"""Durable queue for up to 1,000 pending drawings with bounded CPU concurrency."""
from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path
import sqlite3
import threading
import uuid
from flask import jsonify, request


def register_queue(app, data, process):
    incoming=Path(data)/'incoming';incoming.mkdir(exist_ok=True)
    database=Path(data)/'queue.sqlite'
    lock=threading.Lock()
    executor=ThreadPoolExecutor(max_workers=2,thread_name_prefix='drawing')
    def db():
        c=sqlite3.connect(database,timeout=30);c.row_factory=sqlite3.Row;return c
    with db() as c:
        c.execute('CREATE TABLE IF NOT EXISTS queue(id TEXT PRIMARY KEY,name TEXT,page INTEGER,status TEXT,project_id TEXT,error TEXT,created TEXT DEFAULT CURRENT_TIMESTAMP)')
        c.execute("UPDATE queue SET status='queued' WHERE status='processing'")
    def run(ident):
        with db() as c:
            row=dict(c.execute('SELECT * FROM queue WHERE id=?',(ident,)).fetchone())
            c.execute("UPDATE queue SET status='processing' WHERE id=?",(ident,))
        try:
            with app.test_request_context('/api/projects',method='POST',data={'file':(io.BytesIO((incoming/ident).read_bytes()),row['name']),'page':str(row['page'])}):
                response,status=process()
                if status!=201:raise ValueError('Processing failed')
                project_id=response.get_json()['id']
            with db() as c:c.execute("UPDATE queue SET status='ready',project_id=? WHERE id=?",(project_id,ident))
            (incoming/ident).unlink(missing_ok=True)
        except Exception as e:
            with db() as c:c.execute("UPDATE queue SET status='failed',error=? WHERE id=?",(str(e)[:1000],ident))
    with db() as c:pending=[r['id'] for r in c.execute("SELECT id FROM queue WHERE status='queued'")]
    for ident in pending:executor.submit(run,ident)

    @app.post('/api/queue')
    def enqueue():
        file=request.files.get('file')
        if not file:raise ValueError('Choose a drawing.')
        name=Path(file.filename.replace('\\','/')).name
        if Path(name).suffix.lower() not in ('.pdf','.png','.jpg','.jpeg'):raise ValueError('Supported formats: PDF, PNG, JPEG.')
        try:page=int(request.form.get('page',1))
        except ValueError:raise ValueError('Page must be a positive integer.')
        if page<1:raise ValueError('Page must be a positive integer.')
        ident=uuid.uuid4().hex
        with lock:
            with db() as c:
                active=c.execute("SELECT count(*) FROM queue WHERE status IN ('queued','processing')").fetchone()[0]
                if active>=1000:return jsonify(error='Queue holds 1,000 drawings. Wait for processing to free capacity.'),429
                file.save(incoming/ident)
                c.execute('INSERT INTO queue(id,name,page,status) VALUES(?,?,?,?)',(ident,name,page,'queued'))
            executor.submit(run,ident)
        return jsonify(id=ident,status='queued'),202

    @app.get('/api/queue')
    def queue_status():
        with db() as c:
            counts={r['status']:r['n'] for r in c.execute('SELECT status,count(*) n FROM queue GROUP BY status')}
            rows=[dict(r) for r in c.execute('SELECT * FROM queue ORDER BY created DESC,rowid DESC LIMIT 1000')]
        return jsonify(capacity=1000,workers=2,counts=counts,jobs=rows)
