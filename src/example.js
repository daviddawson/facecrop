const image = require('@canvas/image');
const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const faceapi = require('../model/face-api.node.js');
const sharp = require('sharp');

const modelPath = 'model/';
const ssdOptions = { minConfidence: 0.1, maxResults: 10 };

async function cropFace(inputFile, outputFile, width, height, padding = 0.1) {
  // Read and process the image
  const buffer = fs.readFileSync(inputFile);
  const canvas = await image.imageFromBuffer(buffer);
  const imageData = image.getImageData(canvas);
  console.log('image:', inputFile, canvas.width, canvas.height);

  // Create tensor from image data
  const tensor = tf.tidy(() => {
    const data = tf.tensor(Array.from(imageData?.data || []), [canvas.height, canvas.width, 4], 'int32');
    const channels = tf.split(data, 4, 2);
    const rgb = tf.stack([channels[0], channels[1], channels[2]], 2);
    return tf.reshape(rgb, [1, canvas.height, canvas.width, 3]);
  });
  console.log('tensor:', tensor.shape, tensor.size);

  // Load face detection model
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);

  // Detect faces
  const optionsSSDMobileNet = new faceapi.SsdMobilenetv1Options(ssdOptions);
  const result = await faceapi.detectAllFaces(tensor, optionsSSDMobileNet);

  if (result.length === 0) {
    console.log('No faces detected');
    return;
  }

  // Get the first detected face
  const detection = result[0];
  const box = detection._box;

  // Calculate the extended box with padding
  const paddingWidth = box._width * padding;
  const paddingHeight = box._height * padding;

  const extendedBox = {
    left: Math.max(0, Math.round(box._x - paddingWidth / 2)),
    top: Math.max(0, Math.round(box._y - paddingHeight / 2)),
    width: Math.round(box._width + paddingWidth),
    height: Math.round(box._height + paddingHeight)
  };

  // Ensure the box doesn't exceed image dimensions
  extendedBox.width = Math.min(extendedBox.width, canvas.width - extendedBox.left);
  extendedBox.height = Math.min(extendedBox.height, canvas.height - extendedBox.top);

  // Calculate aspect ratios
  const boxAspectRatio = extendedBox.width / extendedBox.height;
  const targetAspectRatio = width / height;

  // Adjust crop box to maintain aspect ratio
  if (boxAspectRatio > targetAspectRatio) {
    // Box is wider, adjust height
    const newHeight = extendedBox.width / targetAspectRatio;
    const heightDiff = newHeight - extendedBox.height;
    extendedBox.top = Math.max(0, Math.round(extendedBox.top - heightDiff / 2));
    extendedBox.height = Math.round(newHeight);
  } else {
    // Box is taller, adjust width
    const newWidth = extendedBox.height * targetAspectRatio;
    const widthDiff = newWidth - extendedBox.width;
    extendedBox.left = Math.max(0, Math.round(extendedBox.left - widthDiff / 2));
    extendedBox.width = Math.round(newWidth);
  }

  // Ensure the box doesn't exceed image dimensions (again, after aspect ratio adjustment)
  extendedBox.width = Math.min(extendedBox.width, canvas.width - extendedBox.left);
  extendedBox.height = Math.min(extendedBox.height, canvas.height - extendedBox.top);

  // Crop and save the face thumbnail
  await sharp(inputFile)
    .extract(extendedBox)
    .resize(width, height, { fit: 'cover' })
    .toFile(outputFile);

  console.log(`Face thumbnail saved as ${outputFile}`);
}

// Example usage
async function main() {
  const imageFile = './sample1.jpg';
  const outputFile = 'face_thumbnail.jpg';
  const thumbnailWidth = 200;
  const thumbnailHeight = 200;
  const paddingFactor = 0.2;

  await cropFace(imageFile, outputFile, thumbnailWidth, thumbnailHeight, paddingFactor);
}

main();
