import numpy as np
import json
import onnxruntime as rt

model_path = 'model/model.onnx'
class_path = 'model/birds_name_mapping.json'

normalise_means = [0.4914, 0.4822, 0.4465]
normalise_stds = [0.2023, 0.1994, 0.2010]

def normalise_image(image):
    image = image.copy()
    for i in range(3):
        image[:, i, :, :] = (image[:, i, :, :] - normalise_means[i]) / normalise_stds[i]
    return image

def load_class_names():
    with open(class_path, 'r') as f:
        class_names = json.load(f)
    return class_names

def predict(inp_image):
    
    class_names = load_class_names()

    image = inp_image
    image = image.transpose((2, 0, 1))

    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = normalise_image(image)
    image = image.astype(np.float32)

    sess = rt.InferenceSession(model_path)

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    output = sess.run([output_name], {input_name: image})[0]
    prob = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)

    top5 = np.argsort(prob[0])[-5:][::-1]

    class_probs = {class_names[str(i)]: float(prob[0][i]) for i in top5}
    print(class_probs)

    return class_probs