We were given 3 runs of Quark-Gluon Classification data genrated by Impining on a calirometer.
I used the 2nd run only because of storage and computing power limitations.

I trained a VGG 12 model and then a resnet 12 model (because of computing power restraints)
I noticed that it was significantly harder to get an accuracy > ~50%

The resnet model was able to get a Train accuracy of 68% and test accuracy of 52%
The vgg model was able to get a train accuracy of 50% and test accuracy of  49.5%

Both of these models were trained for 50 epochs.

### VGG performance curves
![vgg_acc](https://github.com/fubblyonesandzeros/ML4SciTest2/assets/165657721/ef5d2563-685b-4888-9b35-62ef5b417afd)
![vgg_loss](https://github.com/fubblyonesandzeros/ML4SciTest2/assets/165657721/d20d0903-18af-47cc-b16a-9262ff0492d5)

### Resnet performance curves
![resnet_accuracy](https://github.com/fubblyonesandzeros/ML4SciTest2/assets/165657721/130d901b-4761-44cf-a37f-a6f7b6bc0c17)
![resnet_loss](https://github.com/fubblyonesandzeros/ML4SciTest2/assets/165657721/68aa00b6-e9cb-4b80-9c75-f8259cc84acc)
