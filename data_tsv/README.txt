# data files ready for text classification experiments
# last maintained: 2024-05-12 16:51:51

Each of the data files ready for text classification experiments contains LLM-generated descriptions.
We are confident that the LLM-generated files do not raise any IPR-related issues. 
However, as a precautionary measure, those files are password-protected.

To decode the encoded tar.gz file, please issue the following command in this directory. 
You will need to provide the password, which will be separately notified to you.

$ openssl enc -aes-256-cbc -d -in data_tsv.enc -out data_tsv.tar.gz -pass pass:password

Afterwards, you can expand the tar.gz file to obtain the data files. 
These files will be:
-rw-rw-r-- hayashi/hayashi 1103992 2024-05-07 18:13 ./dev.tsv
-rw-rw-r-- hayashi/hayashi 2403304 2024-05-07 18:13 ./test.tsv
-rw-rw-r-- hayashi/hayashi 9065237 2024-05-07 18:13 ./train.tsv

REMIND that we will immediately remove these files if any IPR-related issues are reported.
Consult me at yoshihiko.hayashi@gmail.com for the password. Thank you.
