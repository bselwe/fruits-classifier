import numpy as np

from keras.models import load_model
from keras.preprocessing import image

def main():
    img = image.load_img("./strawberry3.jpg", target_size=(100, 100))
    x = image.img_to_array(img)
    x /= 255.
    x = np.expand_dims(x, axis=0)

    model = load_model("./output/model_checkpoint.h5")
    res = model.predict(x)

    print(res)

if __name__ == "__main__":
    main()