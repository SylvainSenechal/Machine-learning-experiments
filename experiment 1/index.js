'use strict';
// const tf = require('@tensorflow/tfjs');



const predict = x => {
  return tf.tidy( () => {
    return a.mul(x).add(b)
  })
}

const loss = (predictions, labels) => {
  const meanSquareError = predictions.sub(labels).square().mean()
  return meanSquareError
}


const train = (xs, ys, nbIterations = 200) => {
  const learningRate = 0.1
  const optimizer = tf.train.sgd(learningRate)

  for (let i = 0; i < nbIterations; i++) {
    optimizer.minimize( () => {
      const predsYs = predict(xs)
      return loss(predsYs, ys)
    })
  }
}

const generateData = nbData => {
  return tf.tidy( () => {
    const xs = tf.randomUniform([nbData], 0, 300)
    const a = tf.scalar(Math.random())
    const b = tf.scalar(4)
    const ys = a.mul(xs).add(b)
    return {xs ,ys}
  })
}




const dessin = () => {
	ctx.clearRect(0, 0, canvas.width, canvas.height)

  const ysReal = data.ys.dataSync()
  const ysPredict = predict(data.xs).dataSync()
  console.log(ysPredict)
  console.log(ysReal)
  ctx.strokeStyle = "#000000"
  data.xs.dataSync().forEach( (x, index) => {
    ctx.beginPath()
    ctx.arc(x, ctx.canvas.height*0.5 - ysReal[index], 2, 0, 2 * Math.PI)
    ctx.stroke()
  })
  ctx.strokeStyle = "#ff0000"
  data.xs.dataSync().forEach( (x, index) => {
    ctx.beginPath()
    ctx.arc(x, ctx.canvas.height*0.5 - ysPredict[index], 2, 0, 2 * Math.PI)
    ctx.stroke()
  })

}

const a = tf.variable(tf.scalar(Math.random()))
const b = tf.variable(tf.scalar(Math.random()))
const data = generateData(20)

var ctx, canvas
const initCanvas = () => {
  console.log("oui")
  canvas = document.getElementById('canvas')
  ctx = canvas.getContext('2d')
  ctx.canvas.width = 900
  ctx.canvas.height = 600

  train(data.xs, data.ys)
  // dessin()
  predict(1.0).print()
  a.print()
}
window.addEventListener('load', initCanvas)
