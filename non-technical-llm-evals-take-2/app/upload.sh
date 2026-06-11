#!/usr/bin/env bash

npm run build
rsync -av dist/* tarospec_aisap-test-website@ssh.nyc1.nearlyfreespeech.net:
