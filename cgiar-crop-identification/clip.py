import os
import numpy as np
import cv2
import clip
from PIL import Image
from matplotlib import pyplot as plt




with torch.no_grad():
    for file in files:
        print(file)
        # Load image from file
        img = Image.open(file).convert("RGB")

        # Just show image in the notebook
        plt.imshow(cv2.resize(np.array(img), (256, 256)))
        plt.show()
        
        # Preprocess image using clip
        img = preprocess(img).unsqueeze(0).cuda()
        
        # Get Image embeddings
        image_embeddings = model.encode_image(img)
        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
        
        
        score = []
        for query in QUERIES:
            texts = clip.tokenize(query).cuda()
            
            # Get Text Embeddings
            text_embeddings = model.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            
            # Calc dot product between image and text embeddings
            sc = float((image_embeddings @ text_embeddings.T).cpu().numpy())
            score.append(sc)
        
        print( pd.DataFrame({'query': QUERIES, 'score': score}).sort_values('score', ascending=False) )
        print('')
        print('-------------------------')
        print('')