# LLM-generated descriptions
# last maintained: 2024-05-12 16:42:57

We are confident that the LLM-generated files do not raise any IPR-related issues. 
However, as a precautionary measure, those files are password-protected.

To decode the encoded tar.gz file, please issue the following command in this directory. 
You will need to provide the password, which will be separately notified to you.

$ openssl enc -aes-256-cbc -d -in descriptions.enc -out descriptions.tar.gz -pass pass:password

Afterwards, you can expand the tar.gz file to obtain the LLM-generated description files. 
These files will be:
-rw-rw-r-- 1 hayashi hayashi  279792 Sep 19  2023 wic_desc_dev_contrast_gpt-3.5-turbo-0613.tsv
-rw-rw-r-- 1 hayashi hayashi  252873 Sep 30  2023 wic_desc_dev_contrast_gpt-4-0613.tsv
-rw-rw-r-- 1 hayashi hayashi  241690 Oct  1  2023 wic_desc_dev_direct_gpt-3.5-turbo-0613.tsv
-rw-rw-r-- 1 hayashi hayashi  170545 Sep 30  2023 wic_desc_dev_direct_gpt-4-0613.tsv
-rw-rw-r-- 1 hayashi hayashi  605694 Sep 19  2023 wic_desc_test_contrast_gpt-3.5-turbo-0613.tsv
-rw-rw-r-- 1 hayashi hayashi  551675 Sep 30  2023 wic_desc_test_contrast_gpt-4-0613.tsv
-rw-rw-r-- 1 hayashi hayashi  528082 Oct  1  2023 wic_desc_test_direct_gpt-3.5-turbo-0613.tsv
-rw-rw-r-- 1 hayashi hayashi  372066 Sep 30  2023 wic_desc_test_direct_gpt-4-0613.tsv
-rw-rw-r-- 1 hayashi hayashi 2275935 Sep 19  2023 wic_desc_train_contrast_gpt-3.5-turbo-0613.tsv
-rw-rw-r-- 1 hayashi hayashi 2009532 Oct  1  2023 wic_desc_train_contrast_gpt-4-0613.tsv
-rw-rw-r-- 1 hayashi hayashi 1997710 Oct  1  2023 wic_desc_train_direct_gpt-3.5-turbo-0613.tsv
-rw-rw-r-- 1 hayashi hayashi 1394020 Sep 30  2023 wic_desc_train_direct_gpt-4-0613.tsv

REMIND that we will immediately remove these files if any IPR-related issues are reported.
Consult me at yoshihiko.hayashi@gmail.com for the password. Thank you.
