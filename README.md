# Perception Engines

Working implementation of Perception Engines software that uses machine perception to generate abstract visuals representing categories. For background see my [Perception Engines essay](https://medium.com/artists-and-machine-intelligence/perception-engines-8a46bc598d57) or my more recent paper [Shared Visual Abstactions](https://arxiv.org/abs/1912.04217).

## Getting Started

### Rendering System

Images are generated by "renderers". These generally go into into the
"renderer" subdirectory of the classpath and are loaded dynamically.
They also generally have a simple numbered version scheme since it
is handy to keep old versions around once you have output files
that depend on them.

To see a renderer in action, we can run the "render_images" script.

```bash
python render_images.py
```

This will run the default "lines1" renderer which draws colored lines
on the canvas. With no arguments, the lines will be randomly generated
and then saved to a templated file. Take note of the printed output file
and open it up to see what was created.

![example default output](https://user-images.githubusercontent.com/945979/70071433-42b13980-165a-11ea-80bc-07548473bed7.jpg)


A renderer is really just a python file that contains a render function:
```python
# input: array of real vectors, length 8, each component normalized 0-1
def render(a, size):
```

The numpy array a is a variable list of length 8 vectors and size is
the dimensions of the output image in pixels. The renderer should
generate and return an image. Its very important for the output images
to be identical when size varies. Here are a bunch of different ways to
run `render_images` with more explicit arguments.


Provide the renderer, random seed, size, and size manually
```bash
python render_images.py \
  --renderer lines1 \
  --random-seed 3 \
  --size 600
  ```

![render_seed](https://user-images.githubusercontent.com/945979/70071432-42b13980-165a-11ea-9d6d-bdd14f7b13c4.jpg)

Now that we are supplying a fixed random-seed, we can test if this matches when scaled
```bash
python render_images.py \
  --renderer lines1 \
  --random-seed 3 \
  --size 300
```

![render_seed_small](https://user-images.githubusercontent.com/945979/70071431-42b13980-165a-11ea-847a-d7003d76c168.jpg)

And should change when the random seed is changed
```bash
python render_images.py \
  --renderer lines1 \
  --random-seed 4 \
  --size 300
```

![render_seed_small_r4](https://user-images.githubusercontent.com/945979/70071430-4218a300-165a-11ea-9b08-69a51e5fe285.jpg)

To draw fewer lines, change the length of the input array
```bash
python render_images.py \
  --renderer lines1 \
  --random-seed 4 \
  --length 10 \
  --size 300
```

![render_seed_small_r4_l10](https://user-images.githubusercontent.com/945979/70071429-4218a300-165a-11ea-9697-3cf11ec4532e.jpg)

The output file can be fixed and named with different file formats possible:
```bash
python render_images.py \
  --renderer lines1 \
  --random-seed 4 \
  --length 10 \
  --size 300 \
  --outfile outputs/test_length10.jpg
```

![test_length10](https://user-images.githubusercontent.com/945979/70071428-4218a300-165a-11ea-8282-117b7cb43254.jpg)

Templated output file names using variables are handy. SEQ will auto-increment when re-run. (run this one a few times to get different versions)
```bash
python render_images.py \
  --renderer lines1 \
  --length 10 \
  --size 300 \
  --outfile outputs/test_length10_%SEQ%.jpg
```

![test_length10_10](https://user-images.githubusercontent.com/945979/70071423-41800c80-165a-11ea-8b75-6a819bcb19d8.jpg)
![test_length10_09](https://user-images.githubusercontent.com/945979/70071425-41800c80-165a-11ea-8094-e7f4194f6e8f.jpg)
![test_length10_08](https://user-images.githubusercontent.com/945979/70071427-41800c80-165a-11ea-94a7-d8d2437f617b.jpg)


### Scoring System

There is a separate scoring system currently based on keras pre-trained ImageNet Challenge models.

If you have an image, response graphs can be generated showing topN responses. By default a stock set of 6 ImageNet models will be used, and the output file will be graph_foo.

```bash
python score_images.py \
  --input-glob 'tick.jpg' \
  --target-class tick \
  --do-graphfile
```

![tick graph](https://user-images.githubusercontent.com/945979/69919751-35bf0980-14e5-11ea-9e03-7ead3667d3c7.jpg)

Want to see more graphs? Try all keras imagenet models (currently 18):

```bash
python score_images.py \
  --input-glob 'tick.jpg' \
  --target-class tick \
  --networks all \
  --do-graphfile
```

![tick graph with more networks](https://user-images.githubusercontent.com/945979/69919752-35bf0980-14e5-11ea-8ade-8f0f65805da7.jpg)



### Planning System

Let's get started by drawing a birdhouse.

```bash
python plan_image.py \
  --outdir outputs/birdhouse_1060 \
  --target-class birdhouse \
  --random-seed 1060 \
  --renderer lines1 \
  --num-lines 30
```

This optimizes a drawing to trigger a label of 'birdhouse' on a default set of four
ImageNet models. After several iterations, there will program will end and save a
parameter file `best.npy` in the output directory along with a preview called `best.png`.

![birdhouse1](https://user-images.githubusercontent.com/945979/70126508-0f17f300-16de-11ea-9afa-ee6c083c4960.jpg)

You can run it a few times changing the `outdir` and `random-seed` to get different results.

![birdhouse2](https://user-images.githubusercontent.com/945979/70126505-0f17f300-16de-11ea-83d7-fc6fdb89c083.jpg)
![birdhouse3](https://user-images.githubusercontent.com/945979/70126507-0f17f300-16de-11ea-8d3c-5bbb071a5eb5.jpg)

When you get one you like, you can use the `render_images.py` script to redraw it at higher resolution.

```bash
python render_images.py \
  --input-glob 'outputs/birdhouse_1080/best.npy' \
  --outbase best_1920.jpg \
  --renderer lines1 \
  --size 1920
```

![best_960](https://user-images.githubusercontent.com/945979/70126694-6f0e9980-16de-11ea-96ae-db1f80cc58a2.jpg)

Here we use `input-glob` to provide the inputs (wildcards are allowed), and instead
of outfile we use `outbase` which saves the named file in the same directory location
as the input file.

How well does this result generalize to other networks? To test that we can run on all
ImageNet networks. It's also helpful to highligh the four networks which were used in
"training" this image, and that group has the nickname "train1".

```bash
python score_images.py \
  --input-glob 'outputs/birdhouse_1080/best_1920.jpg' \
  --train1 train1 \
  --target-class birdhouse \
  --networks all \
  --do-graphfile
```

![graph_best_960](https://user-images.githubusercontent.com/945979/70126831-aa10cd00-16de-11ea-9250-b9c357b9f182.jpg)

Wow - this result generalizes really well to other network architectures. The first networks in yellow were used to make this image, but all of the other networks also give strong top1 results. But does this result also generalize to other training sets?

If you have google vision and aws credentials setup correctly you can additionaly test this image against their public APIs (and specify the target label). And here we also specify the `graphfile-prefix` explicitly which changes the output filename.

```bash
python score_images.py \
  --input-glob 'outputs/birdhouse_1080/best_1920.jpg' \
  --train1 train1 \
  --networks train1,aws:+birdhouse,goog:+birdhouse \
  --target-class birdhouse \
  --graphfile-prefix graph_apis_ \
  --do-graphfile
```

![graph_apis_best_1920.jpg](https://user-images.githubusercontent.com/945979/70133620-5d7fbe80-16eb-11ea-85ab-2e69b0523df5.jpg)

The google vision results seem to have nothing to do with birdhouses, just labels for things like `illustration` and `clip art`. The amazon rekognition results are also not showing an exact match for `birdhouse`, though reading the tea leaves we do see there are top5 results for `building` and the more specific label `bird feeder` - both of which seem like neighboring concepts.


