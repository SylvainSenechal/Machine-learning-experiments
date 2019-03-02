///////////////////////////////////////////////////////////////////////////////
// BASED ON https://github.com/tensorflow/tfjs-examples/tree/master/mnist-acgan
///////////////////////////////////////////////////////////////////////////////

'use strict';

const IMAGE_SIZE = 28
const LR = 0.0002
const ADAM_BETA = 0.5

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

    const realness = tf.layers.dense({units: 1, activation: 'softmax'}).apply(features) // voir sigmoid
    return tf.model({inputs: image, outputs: realness})
  }
}

const trainDiscriminator = () => {
  // const realX = tf.from array ..

  // const realX = tf.randomUniform([IMAGE_SIZE, IMAGE_SIZE], 0, 1)
  // const fakeX = tf.randomUniform([IMAGE_SIZE, IMAGE_SIZE], 0, 1)
  //
  // const x = tf.concat([realX, fakeX], 0)
  // const y = tf.tensor([1, 0]) // yreal yfake
  const x = tf.randomUniform([IMAGE_SIZE, IMAGE_SIZE], 0, 1).reshape([1, 28, 28, 1])
  const y = tf.tensor([1]) // yreal yfake


  x.print()
  y.print()

  const loss = discriminator.cnn.trainOnBatch(x, y)
  return loss
}

const run = () => {
  discriminator = new Discriminator()
  discriminator.cnn.compile({
    optimizer: tf.train.adam(LR, ADAM_BETA),
    loss: ['binaryCrossentropy']
  })
  discriminator.cnn.summary()

  for (let i = 0; i < 2; i++) {
    const discriminatorLoss = trainDiscriminator()
  }
}

let discriminator
run()
var ctx, canvas
const initCanvas = () => {
  canvas = document.getElementById('canvas')
  ctx = canvas.getContext('2d')

  ctx.canvas.width = window.innerWidth
  ctx.canvas.height = window.innerHeight

  canvas.addEventListener('mousemove', mousemove)
  canvas.addEventListener('mousedown', mouseDown)
  canvas.addEventListener('mouseup', mouseUp)
}
let mouseDrawing = false
const mouseDown = mouse => mouseDrawing = true
const mouseUp = mouse => mouseDrawing = false

const click = mouse => {
  if (game.mouseAction === "attack") {
    game.listEnnemi.forEach( (e, index, list) => {
      if( dst(e.x, e.y, mouse.offsetX, mouse.offsetY) < 2*e.size){
        e.life -= game.clickDamage
      }
    })
  }
  else if (game.mouseAction === "archer") {
    game.listWorkers.push(new Archer(mouse.offsetX, mouse.offsetY))
  }
}
let grid = Array(28*28).fill(0)//.map( () => Array(28).fill(0) )
console.log(grid)
const mousemove = mouse => {
  if (mouseDrawing === true) {
    let row = Math.floor(mouse.clientX/5)
    let col = Math.floor(mouse.clientY/5)
    try {
      grid[row * 28 + col] = 1
      drawGrid()
    }
    catch(err) {
      // console.log(err)
    }
  }
}

const drawGrid = () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height)

  grid.forEach( (value, id) => {
    if (value === 0) ctx.fillStyle = '#000000'
    else if (value === 1) ctx.fillStyle = '#ffffff'
    ctx.fillRect(id*5, id*5, 5, 5)
  })
  // grid.forEach( (row, idRow) => {
  //   row.forEach( (value, idCol) => {
  //     if (value === 0) ctx.fillStyle = '#000000'
  //     else if (value === 1) ctx.fillStyle = '#ffffff'
  //     ctx.fillRect(idRow*5, idCol*5, 5, 5)
  //   })
  // })
}

// let arr1 = [[1,2,3], [4,5,6]]
// let arr = [1,2,3,4,5,6]
// let tarr = tf.tensor3d(arr, [2, 3, 1])
// tarr.print()


//
// function dessin(){
// 	ctx.clearRect(0, 0, canvas.width, canvas.height)
//   for(let i=0; i<Partie.nbCase; i++){
//     for(let j=0; j<Partie.nbCase; j++){
//       if(Partie.grid[i][j] === 0){
//         ctx.fillStyle = "#000000"
//       }
//       else{
//         ctx.fillStyle = 'rgb(' +Partie.grid[i][j]*255+ ',' +Partie.grid[i][j]*255+ ',' +Partie.grid[i][j]*255+ ')'
//       }
//       ctx.fillRect(i*Partie.widthCase+Partie.offset, j*Partie.widthCase+Partie.offset, Partie.widthCase, Partie.widthCase);
//     }
//   }
// }


window.addEventListener('load', initCanvas)




















// let generator = new Generator()
// console.log(generator)
//
//
// class Generator {
//   constructor(zSpace) {
//     this.cnn = this.createCnnModel()
//     this.cnn.summary()
//   }
//   createCnnModel = () => {
//     let cnn = tf.sequential()
//
//     return cnn
//   }
// }
