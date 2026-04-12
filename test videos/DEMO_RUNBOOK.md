# Demo Runbook for React App

Use these videos in this order during your presentation:

1. Real sample (expected real/low risk)
   - demo_real_01.mp4
   - demo_real_02.mp4
   - demo_real_03.mp4

2. AI medium confidence sample (expected fake with medium confidence)
   - demo_ai_medium_confidence.mp4
   - Source: data/raw/fake/ffc23_deepfakedetection_01_04__talking_angry_couch__0XUW13RW.mp4
   - Local pipeline reading used for curation: fake probability ~0.985

3. AI hard sample (expected fake but harder case)
   - demo_ai_hard_case.mp4
   - Source: data/raw/fake/ffc23_deepfakedetection_01_03__talking_against_wall__JZUXXFRB.mp4
   - Local pipeline reading used for curation: fake probability ~0.946

Reference mapping file:
- DEMO_VIDEO_MANIFEST.csv

How to run React demo:
1. Start backend from project root:
   - python api_server.py
2. Start frontend from frontend folder:
   - npm run dev
3. Open frontend URL and upload the videos above one-by-one.

Presentation tip:
- Show one real sample first, then the medium AI sample, then the hard AI sample to demonstrate confidence variation.
