# preprocessing parameters were taken from
# https://github.com/NVlabs/edm2/blob/4bf8162f601bcc09472ce8a32dd0cbe8889dc8fc/README.md#preparing-datasets

python3 _preprocess.py convert \
    --source=imagenet-ILSVRC2012-validation \
    --dest=preprocessed/ \
    --resolution=512x512 \
    --transform=center-crop-dhariwal

# move images to `preprocessed-images` folder
mkdir preprocessed-images

for $f in $(ls preprocessed-images/); do
    mv preprocessed/$f/* preprocessed-images/
done

# rm the empty folder
rm -r preprocessed
