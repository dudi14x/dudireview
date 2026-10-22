import re
import pytest
from floorreview.app import create_app

@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setenv('FLOORREVIEW_MODEL',str(tmp_path/'missing.joblib'))
    app=create_app(tmp_path);app.testing=True
    client=app.test_client()
    token=re.search("window.FR_TOKEN='([^']+)'",client.get('/').text)[1]
    return client,{'X-FloorReview-Token':token}
