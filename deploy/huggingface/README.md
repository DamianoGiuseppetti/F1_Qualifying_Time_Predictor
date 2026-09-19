---
title: F1 Qualifying Predictor
emoji: 🏎️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
suggested_hardware: cpu-basic
pinned: false
short_description: LSTM qualifying-time predictions for the 2026 F1 season
---

This Space runs the F1 Qualifying Predictor app: an LSTM model that
predicts each driver's qualifying time from practice-session pace, with
an uncertainty interval around every prediction.

Full project write-up, results, and how to reproduce it from scratch:
https://github.com/DamianoGiuseppetti/F1_Qualifying_Time_Predictor

This README is Hugging Face's own Space metadata file (the YAML block
above is required by the platform), not the project README - it is
never meant to be read on GitHub. See deploy/huggingface/SETUP.md in the
main repo for how this Space gets built and updated.
