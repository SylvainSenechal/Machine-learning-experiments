"use strict";
const path = require('path');
const fs = require("fs");
const tf = require('@tensorflow/tfjs');

const getXsTrainData = () => {
  let xs = fs.readFileSync(path.join(__dirname, 'train.csv'), 'utf-8')
    .split(/\r?\n/) // Separation des lignes
    .map( line => line.split(',').splice(2)) // vire id et target : splice 2
    // .map( line => line.split(',').splice(282))
  xs.pop()
  xs.shift()
  return xs
}
const getYsTrainData = () => {
  let ys = fs.readFileSync(path.join(__dirname, 'train.csv'), 'utf-8')
    .split(/\r?\n/) // Separation des lignes
    .map( line => line.split(',').splice(1, 1))
  ys.pop()
  ys.shift()
  return ys.flat()
}

const getXsTestData = () => {
  let xs = fs.readFileSync(path.join(__dirname, 'test.csv'), 'utf-8')
    .split(/\r?\n/) // Separation des lignes
    .map( line => line.split(',').splice(1)) // vire id : splice 1
    // .map( line => line.split(',').splice(281))
  xs.pop()
  xs.shift()
  return xs
}

const predict = xs => {
  return tf.tidy( () => {
    return weights.mul(xs).sum(1).add(biase)
  })
}

const train = (xs, ys, nbIterations = 300) => {
  const learningRate = 0.01
  const optimizer = tf.train.sgd(learningRate)

  for (let i = 0; i < nbIterations; i++) {
    optimizer.minimize( () => {
      const ysPredictions = predict(xs)
      return loss(ysPredictions, ys)
    })
  }
}

const loss = (ysFake, ysReal) => {
  const meanSquareError = ysFake.sub(ysReal).square().mean()
  return meanSquareError
}


const xs = tf.tensor2d(getXsTrainData(), [250, 300], 'float32') // 250 lignes de 300 elements
const ys = tf.tensor1d(getYsTrainData(), 'float32') // pas de shape en 1d
const weights = tf.variable(tf.randomUniform([300], 0, 1))
const biase = tf.variable(tf.scalar(Math.random()))


const accuracy = (xsToPredict, ysReal) => {
  let ysPredicted = predict(xsToPredict).dataSync().map( e => e > 0 ? 1:0)
  let ys = ysReal.dataSync()

  let goodGuess = 0
  for (let i = 0; i < ys.length; i++) {
    if (ysPredicted[i] === ys[i]) goodGuess++
  }
  console.log("Good guess : ", goodGuess, " out of ", ys.length)
}

const predictTest = xsTest => {
  let ysPredicted = predict(xsTest).dataSync().map( e => e > 0 ? 1:0)
  console.log(ysPredicted.length)

  let file = fs.createWriteStream('results.csv')
  file.on('error', err => console.log(err))
  file.write('id,target\n')
  ysPredicted.forEach( (prediction, index) => {
    let line = (250+index).toString() + "," + prediction + "\n"
    file.write(line)
  })
  file.end()
}

biase.print()
train(xs, ys)
accuracy(xs, ys)
biase.print()
const xsTest = tf.tensor2d(getXsTestData(), [19750, 300], 'float32') // 19750 lignes de 300 elements
predictTest(xsTest)
