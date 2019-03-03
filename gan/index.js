///////////////////////////////////////////////////////////////////////////////
// BASED ON https://github.com/tensorflow/tfjs-examples/tree/master/mnist-acgan
///////////////////////////////////////////////////////////////////////////////

'use strict';

const IMAGE_SIZE = 28
const LR = 0.0002
const ADAM_BETA = 0.5
const LATENT_SIZE = 10

let discriminator, generator, combined

let ctxTrain, canvasTrain
let ctxPredict, canvasPredict
let ctxGen, canvasGen
let gridTrain = Array(28).fill().map( () => Array(28).fill(0) )
let gridPredict = Array(28).fill().map( () => Array(28).fill(0) )
let gridGen = Array(IMAGE_SIZE*IMAGE_SIZE).fill().map( () => Math.random() )
const CASE_SIZE = 5
let mouseDrawing = false

class Generator {
  constructor() {
    this.cnn = this.createCnnModel()
  }

  createCnnModel() {
    let cnn = tf.sequential()
    cnn.add(tf.layers.dense({
      units: 3 * 3 * 384,
      inputShape: [LATENT_SIZE],
      activation: 'relu'
    }))
    cnn.add(tf.layers.reshape({targetShape: [3, 3, 384]}))

    cnn.add(tf.layers.conv2dTranspose({
      filters: 192,
      kernelSize: 5,
      strides: 1,
      padding: 'valid',
      activation: 'relu',
      kernelInitializer: 'glorotNormal'
    }))
    cnn.add(tf.layers.batchNormalization())

    cnn.add(tf.layers.conv2dTranspose({
      filters: 96,
      kernelSize: 5,
      strides: 2,
      padding: 'same',
      activation: 'relu',
      kernelInitializer: 'glorotNormal'
    }))
    cnn.add(tf.layers.batchNormalization())

    cnn.add(tf.layers.conv2dTranspose({
      filters: 1,
      kernelSize: 5,
      strides: 2,
      padding: 'same',
      activation: 'tanh',
      kernelInitializer: 'glorotNormal'
    }))

    const latent = tf.input({shape: [LATENT_SIZE]})

    const fakeImg = cnn.apply(latent)
    return tf.model({inputs: latent, outputs: fakeImg})
  }
}

class Discriminator {
  constructor() {
    this.cnn = this.createCnnModel()
  }

  createCnnModel() {
    let cnn = tf.sequential()

    cnn.add(tf.layers.conv2d({
      filters: 32,
      kernelSize: 3,
      padding: 'same',
      strides: 2,
      inputShape: [IMAGE_SIZE, IMAGE_SIZE, 1]
    }))
    cnn.add(tf.layers.leakyReLU({alpha: 0.2}))
    cnn.add(tf.layers.dropout({rate: 0.3}))

    cnn.add(tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      padding: 'same',
      strides: 1,
    }))
    cnn.add(tf.layers.leakyReLU({alpha: 0.2}))
    cnn.add(tf.layers.dropout({rate: 0.3}))

    cnn.add(tf.layers.conv2d({
      filters: 128,
      kernelSize: 3,
      padding: 'same',
      strides: 2,
    }))
    cnn.add(tf.layers.leakyReLU({alpha: 0.2}))
    cnn.add(tf.layers.dropout({rate: 0.3}))

    cnn.add(tf.layers.conv2d({
      filters: 256,
      kernelSize: 3,
      padding: 'same',
      strides: 1,
    }))
    cnn.add(tf.layers.leakyReLU({alpha: 0.2}))
    cnn.add(tf.layers.dropout({rate: 0.3}))

    cnn.add(tf.layers.flatten())

    const image = tf.input({shape: [IMAGE_SIZE, IMAGE_SIZE, 1]})
    const features = cnn.apply(image)

    const realness = tf.layers.dense({units: 1, activation: 'sigmoid'}).apply(features) // voir sigmoid vs softmax
    return tf.model({inputs: image, outputs: realness})
  }
}

const trainDiscriminator = async () => {
  const realX = tf.tensor(gridTrain.flat(), [1, 28, 28, 1])
  // const fakeX = tf.randomUniform([IMAGE_SIZE, IMAGE_SIZE], 0, 1).reshape([1, 28, 28, 1])
  const zVector = tf.randomUniform([1, LATENT_SIZE], -1, 1)
  const fakeX = generator.cnn.predict([zVector])
  const x = tf.concat([realX, fakeX])
  const y = tf.tensor([1, 0])

  const loss = await discriminator.cnn.trainOnBatch(x, y)
  return loss
}

const trainGenerator = async () => {
  const zVector = tf.randomUniform([1, LATENT_SIZE], -1, 1)
  const y = tf.tensor([1]) // faire croise que l image generee est vraie

  const loss = combined.trainOnBatch(zVector, y)
  return loss
}

const run = async () => {
  discriminator = new Discriminator()
  discriminator.cnn.compile({
    optimizer: tf.train.adam(LR, ADAM_BETA),
    loss: ['binaryCrossentropy']
  })
  discriminator.cnn.summary()

  generator = new Generator()
  generator.cnn.summary()

  // discriminator.trainable = false
  const latent = tf.input({shape: [LATENT_SIZE]})
  let fake = generator.cnn.apply([latent])

  fake = discriminator.cnn.apply(fake)

  // Explications :
  // Le modele combined prend en entree le z space
  // et il retourne fake, c est a dire :
  // une image generee par generator, feed au discriminator qui nous dit si fake ou non (bool)
  combined = tf.model({
    inputs: latent,
    outputs: fake
  })
  combined.compile({
    optimizer: tf.train.adam(LR, ADAM_BETA),
    loss: ['binaryCrossentropy']
  })
  combined.summary()

  for (let i = 0; i < 200; i++) {
    const discriminatorLoss = await trainDiscriminator()
    const generatorLoss = await trainGenerator()

    console.log("dLoss : ", discriminatorLoss)
    console.log("gLoss : ", generatorLoss, "\n\n")
  }
  discriminator.cnn.predict(tf.tensor(gridPredict.flat(), [1, 28, 28, 1])).print()
  discriminator.cnn.predict(tf.randomUniform([IMAGE_SIZE, IMAGE_SIZE], 0, 1).reshape([1, 28, 28, 1])).print()
}

document.addEventListener('keydown', e => {
  if (e.keyCode === 84) run() // t -> train
  else if (e.keyCode === 71) { // g -> generate
    gridGen = generator.cnn.predict(tf.randomUniform([1, LATENT_SIZE], -1, 1)).dataSync()
    console.log(gridGen)
  }
  else {
    discriminator.cnn.predict(tf.tensor(gridPredict.flat(), [1, 28, 28, 1])).print()
  }
  // gridTrain = gridTrain.map(e => e.map(ee => 0))
})
///////////////////
// CANVAS FUNCTIONS

const initCanvas = () => {
  canvasTrain = document.getElementById('canvasTrain')
  ctxTrain = canvasTrain.getContext('2d')
  ctxTrain.canvas.width = CASE_SIZE * IMAGE_SIZE
  ctxTrain.canvas.height = CASE_SIZE * IMAGE_SIZE

  canvasPredict = document.getElementById('canvasPredict')
  ctxPredict = canvasPredict.getContext('2d')
  ctxPredict.canvas.width = CASE_SIZE * IMAGE_SIZE
  ctxPredict.canvas.height = CASE_SIZE * IMAGE_SIZE

  canvasGen = document.getElementById('canvasGen')
  ctxGen = canvasGen.getContext('2d')
  ctxGen.canvas.width = CASE_SIZE * IMAGE_SIZE
  ctxGen.canvas.height = CASE_SIZE * IMAGE_SIZE

  canvasTrain.addEventListener('mousemove', mousemove)
  canvasPredict.addEventListener('mousemove', mousemove)
  document.addEventListener('mousedown', mouseDown)
  document.addEventListener('mouseup', mouseUp)

  drawGridTrain()
  drawGridPredict()
  drawGridGen()
}


// A clean : faire 2 canvas avec 2 listener + 2 fonctions draw differentes
const mousemove = mouse => {
  if (mouseDrawing === true) {
    // Drawing in train grid
    if (mouse.target.id === "canvasTrain") {
      let rowTrain = Math.floor(mouse.layerY / CASE_SIZE)
      let colTrain = Math.floor(mouse.layerX / CASE_SIZE)

      gridTrain[rowTrain][colTrain] = 1
      drawGridTrain()
    }

    // Drawing in predict grid
    if (mouse.target.id === "canvasPredict") {
      let rowPredict = Math.floor(mouse.layerY / CASE_SIZE)
      let colPredict = Math.floor(mouse.layerX / CASE_SIZE)

      gridPredict[rowPredict][colPredict] = 1
      drawGridPredict()
    }
  }
}
const mouseDown = mouse => mouseDrawing = true
const mouseUp = mouse => mouseDrawing = false

const drawGridTrain = () => {
  ctxTrain.clearRect(0, 0, canvasTrain.width, canvasTrain.height)

  gridTrain.forEach( (row, idRow) => {
    row.forEach( (value, idCol) => {
      if (value === 0) ctxTrain.fillStyle = '#000000'
      else if (value === 1) ctxTrain.fillStyle = '#ffffff'
      ctxTrain.fillRect(idCol*CASE_SIZE, idRow*CASE_SIZE, CASE_SIZE, CASE_SIZE)
    })
  })
}

const drawGridPredict = () => {
  ctxPredict.clearRect(0, 0, canvasPredict.width, canvasPredict.height)

  gridPredict.forEach( (row, idRow) => {
    row.forEach( (value, idCol) => {
      if (value === 0) ctxPredict.fillStyle = '#000000'
      else if (value === 1) ctxPredict.fillStyle = '#ffffff'
      ctxPredict.fillRect(idCol*CASE_SIZE, idRow*CASE_SIZE, CASE_SIZE, CASE_SIZE)
    })
  })
}

const drawGridGen = () => {
  ctxGen.clearRect(0, 0, canvasGen.width, canvasGen.height)

  gridGen.forEach( (value, id) => {
    ctxGen.fillStyle = `rgb(${value*255}, ${value*255}, ${value*255})`
    let x = id % IMAGE_SIZE
    let y = Math.floor(id / IMAGE_SIZE)
    ctxGen.fillRect(x*CASE_SIZE, y*CASE_SIZE, CASE_SIZE, CASE_SIZE)
  })
}

window.addEventListener('load', initCanvas)
// window.addEventListener('load', run)
