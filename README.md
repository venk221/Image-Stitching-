For `Phase 1` we have a jupyter notebook with the name `Phase1_notebook.ipynb`. Our algorithm works best with stitching 2 raw images. Thus two images are read and stitched together in this notebook



2) To run the Phase 2 code:

i) For Supervised Training- Run `python3 Train_custom.py --NumEpochs=50 --MiniBatchSize=64`. This will run the training process for the supervised part. This will generate a Training loss graph -'Train_loss.png'
ii) To test the trained network - run `Test_custom.py`. This will generate an image and loss.
iii) For the Unsupervised part- Run `python3 Train_custom_unsupervised.py`. This will generate a Training loss graph- 'Train_loss_unsupervised.png'
iv) To test the trained unsupervised network - run `Test_custom_unsupervised.py`. 

Checkpoints for Supervised and Unsupervised Training is stored in different checkpoint folders.

Put the Test Set in the root folder of this directory. That is it should be at the same level as the `Phase1` and `Phase2` folder.

Please create a different `Checkpoints_unsupervised` folder in the `Phase 2` directory.
