if [ ! -f config.json ]; then
  echo "WARNING: No config.json found - deploying without a default API key." >&2
  echo "         Whatever config.json is already on the server (if any) stays active." >&2
  echo "         To set a default key: cp config.json.example config.json and edit it." >&2
fi
npm run build
rsync -av dist/* tarospec_aisap-test-website@ssh.nyc1.nearlyfreespeech.net:
if [ -f config.json ]; then
  rsync -av config.json tarospec_aisap-test-website@ssh.nyc1.nearlyfreespeech.net:
fi
