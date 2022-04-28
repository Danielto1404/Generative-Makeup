from stylegan.utils import load_gan, generate_image

if __name__ == '__main__':
    print("loading")
    G = load_gan('../../../Documents/make-me-up/server/static/sg3-make-faces.pkl')
    print("generating..")
    img = generate_image(G)
    print("generated!")
    print(img)
