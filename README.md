# CNN_for_text
## wpcnn example
* input shape: [None, 106, 128, 1]
* the size of filters: 2 (for bigram)
* the number of filters: 32

```
from wpcnn import Weighted_CNN

inputs = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, DEPTH])
wpcnn1=Weighted_CNN(incoming=inputs, input_shape=[HEIGHT, WIDTH, DEPTH], fsize=2, fnumber=32)
outputs=wpcnn1.output
L2=wpcnn1.L2
```
