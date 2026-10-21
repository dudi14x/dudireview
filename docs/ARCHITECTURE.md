# Architecture

```mermaid
flowchart TD
    A[Up to 1000 drawings per batch] --> B[Durable SQLite queue]
    B --> C[Two bounded workers]
    C --> D[PDF/image decoding and high resolution render]
    D --> E[Overlapping YOLO room instance segmentation]
    D --> F[Automatic floor outline proposal]
    D --> G[Embedded PDF labels and unit markers]
    E --> H[Validated polygons]
    F --> H
    G --> I[Provisional space classification]
    H --> J[Calibrated metric area engine]
    I --> J
    J --> K[Dashboard and review]
    K --> L[JSON CSV and semantic label exports]
    L --> M[Supervised training and held-out validation]
    M --> N[Explicit checkpoint selection]
    N --> E
```

The Flask/Waitress process owns a thread-safe inference cache. YOLO inference is serialized because its predictor is mutable; the queue has two workers so rendering and IO can overlap. This is a single-process reference service. Do not start multiple app processes against the same queue without adding atomic job leasing and a shared worker broker.

Queue state, source uploads awaiting processing and finished JSON/images are on disk. Restart requeues interrupted jobs; a process crash between result creation and queue completion can create a duplicate result on retry. No exactly-once claim is made. Versioned result edits reject stale revisions and write through an atomic replace.

The frontend polls queue state and opens results on demand. It does not load 1000 full plan images into the browser. Queue admission is capped at 1000 pending/processing entries and the last 1000 jobs are listed. CPU throughput depends on page size and tile count; no 1000-page inference benchmark has been performed.

No external API, paid inference service, OCR service or credentials are needed for the current pipeline. GitHub is used to retrieve the pinned pretrained checkpoint; Python package indexes supply dependencies. Deployment as a shared/public service requires authentication, per-user data isolation, worker leasing, retention/storage quotas, monitoring and operational load testing.

## Optional Gemini review

The local trained vision detector still runs first. Each phase can request a separate visual review from Google's `generateContent` endpoint. No API call is made by file upload or phase navigation. The server sends a bounded annotated overview and detail crops, validates a structured JSON response for known region IDs, and stores it in a separate SQLite cache. A transactional daily-attempt budget bounds requests. API keys are supplied server-side by environment or an ignored UTF-8 `.env.local` file.

Gemini cannot submit boundaries, dimensions, scale or area values. Suggestions are stored separately and only selected label/kind/occupancy updates can be applied. Application checks revision and full drawing-state fingerprints to reject stale/cross-project suggestions. Applied suggestions clear the review flag; they do not become training ground truth automatically. Model confidence is self-reported and uncalibrated. Missing detections and inaccurate geometry remain limitations of the local detector.
