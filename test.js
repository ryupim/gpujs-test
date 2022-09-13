const { GPU } = require('gpu.js');

const num = 2048;

const generateMatrices = () => {
  const matrices = [[], []]
  for (let y = 0; y < num; y++){
    matrices[0].push([])
    matrices[1].push([])
    for (let x = 0; x < num; x++){
      matrices[0][y].push(Math.random())
      matrices[1][y].push(Math.random())
    }
  }
  return matrices
}

const matrices = generateMatrices()

// using CPU
const cpu = new GPU({ mode: 'cpu', });
const multiplyMatrixCPU = cpu.createKernel(function(a, b) {
  let sum = 0;
  for (let i = 0; i < 2048; i++) {
    sum += a[this.thread.y][i] * b[i][this.thread.x];
  }
  return sum;
}).setOutput([num, num])

const start1 = performance.now();
let out = multiplyMatrixCPU(matrices[0], matrices[1])
const end1 = performance.now();
console.log(out[10][12]) // Logs the element at the 10th row and the 12th column of the output matrix
console.log("cpu process time: ", end1 - start1);

// using GPU
const gpu = new GPU();
const multiplyMatrixGPU = gpu.createKernel(function(a, b) {
  let sum = 0;
  for (let i = 0; i < 1024; i++) {
    sum += a[this.thread.y][i] * b[i][this.thread.x];
  }
  return sum;
}).setOutput([num, num])

const start2 = performance.now();
out = multiplyMatrixGPU(matrices[0], matrices[1])
const end2 = performance.now();

console.log(out[10][12]) // Logs the element at the 10th row and the 12th column of the output matrix
console.log("gpu process time: ", end2 - start2);
