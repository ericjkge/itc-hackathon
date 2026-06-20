# GPU fixture generation

The deployed site is static and cache-only. This backend exists solely to
regenerate the committed responses in `src/agenthn/webapp/static/fixtures/`.
It binds to localhost and does not create a tunnel or change frontend config.

## Capture missing Large memory runs

```bash
CAPTURE_FIXTURES=1 CAPTURE_SIZES=large bash scripts/run_demo_server.sh
```

The capture script records all three Large scenarios, refreshes the recording
inventory in `memory_meta.json`, and updates `.captured.json`. Large remains
disabled in the browser until those three fixture files exist.

To recapture existing files, add `CAPTURE_FORCE=1`. To run the fixture backend
without capturing, omit `CAPTURE_FIXTURES` and stop it with Ctrl-C.
