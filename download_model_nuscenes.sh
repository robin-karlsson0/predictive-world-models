#!/bin/bash

# Ref: https://bcrf.biochem.wisc.edu/2021/02/05/download-google-drive-files-using-wget/

# Model
fileid=1PJl2q0JQDo1FRHwZ0UG_zyVlpMTDmh3t
link='https://docs.google.com/uc?export=download&'id=$fileid
filename=pred_wm_model_nuscenes.th
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate $link -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$fileid" -O $filename && rm -rf /tmp/cookies.txt

# Model EMA
fileid=1p9G0hT5L0Yovj7Moh4UMPkr-nyK8lamB
link='https://docs.google.com/uc?export=download&'id=$fileid
filename=pred_wm_model_ema_nuscenes.th
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate $link -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$fileid" -O $filename && rm -rf /tmp/cookies.txt

# Model optimizer
# fileid=1rSHj8bLh2dEA9_oNY4DwGHec54VDgRNE
# link='https://docs.google.com/uc?export=download&'id=$fileid
# filename=pred_wm_model_opt_nuscenes.th
# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate $link -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$fileid" -O $filename && rm -rf /tmp/cookies.txt

# Test sample
fileid=1ey4-nONreuOdaNFSOfJCvIn9VcmgDIyY
link='https://docs.google.com/uc?export=download&'id=$fileid
filename=test_sample_nuscenes.pkl.gz
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate $link -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$fileid" -O $filename && rm -rf /tmp/cookies.txt
fileid=1ZlJQrqFGfkljmV3ud3EWfcuWg58CRk29
link='https://docs.google.com/uc?export=download&'id=$fileid
filename=test_sample_nuscenes.png
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate $link -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$fileid" -O $filename && rm -rf /tmp/cookies.txt
