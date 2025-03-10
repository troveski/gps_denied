import cv2
import numpy as np
import glob
import os

# --------- Configurações ---------
query_path = 'marcus.png'      # Imagem de consulta
folder_path = 'roma_tiles'           # Pasta onde estão as imagens
output_folder = 'matching_results'  # Pasta para salvar os resultados

# Cria o folder de saída, se não existir
os.makedirs(output_folder, exist_ok=True)

# --------- Carregar e Processar a Imagem de Consulta ---------
query_img = cv2.imread(query_path)
if query_img is None:
    raise ValueError("Imagem de consulta não encontrada: " + query_path)
query_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)

# Inicializa o detector ORB com até 5000 features
orb = cv2.ORB_create(nfeatures=5000)
kp_query, des_query = orb.detectAndCompute(query_gray, None)
print("Keypoints na imagem de consulta:", len(kp_query))

# --------- Configuração do Matcher ---------
# Usamos BFMatcher com norma Hamming e crossCheck para obter matches mais robustos
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# --------- Coleta as Imagens do Folder ---------
# Procura por arquivos com extensão PNG, JPG e JPEG
image_files = glob.glob(os.path.join(folder_path, '*.png')) + \
              glob.glob(os.path.join(folder_path, '*.jpg')) + \
              glob.glob(os.path.join(folder_path, '*.jpeg'))
print(f"Encontradas {len(image_files)} imagens no folder '{folder_path}'.")

# Variáveis para armazenar a melhor imagem (com maior número de matches)
best_match_count = 0
best_image = None
best_matches = None
best_kp = None
best_image_name = ""

# --------- Itera sobre cada imagem do folder para matching ---------
for img_path in image_files:
    img = cv2.imread(img_path)
    if img is None:
        print("Erro ao carregar a imagem:", img_path)
        continue
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp_img, des_img = orb.detectAndCompute(img_gray, None)
    print("Processando {}: {} keypoints encontrados.".format(os.path.basename(img_path), len(kp_img)))
    
    if des_img is None:
        continue
    
    # Realiza o matching entre a imagem de consulta e a imagem atual
    matches = bf.match(des_query, des_img)
    # Ordena os matches pela distância (quanto menor, melhor)
    matches = sorted(matches, key=lambda x: x.distance)
    
    print("Matches encontrados em {}: {}".format(os.path.basename(img_path), len(matches)))
    
    # Desenha os 20 melhores matches (ou menos, se não houver 20)
    drawn_matches = matches[:20]
    img_matches = cv2.drawMatches(query_img, kp_query, img, kp_img, drawn_matches, None, flags=2)
    
    # Salva a imagem com os matches no folder de saída
    save_path = os.path.join(output_folder, "matches_" + os.path.basename(img_path))
    cv2.imwrite(save_path, img_matches)
    print("Imagem de matching salva em:", save_path)
    
    # Exibe a imagem com os matches por 500 ms (opcional)
    #cv2.imshow("Matches: " + os.path.basename(img_path), img_matches)
    #cv2.waitKey(500)
    #cv2.destroyWindow("Matches: " + os.path.basename(img_path))
    
    # Atualiza a melhor imagem se esta tiver mais matches
    if len(matches) > best_match_count:
        best_match_count = len(matches)
        best_image = img.copy()
        best_matches = matches
        best_kp = kp_img
        best_image_name = os.path.basename(img_path)

cv2.destroyAllWindows()

# --------- Exibição e Salvamento do Melhor Matching ---------
if best_image is None:
    print("Nenhum matching satisfatório foi encontrado em nenhuma imagem do folder.")
else:
    print("Melhor imagem encontrada:", best_image_name)
    print("Número de matches:", best_match_count)
    # Desenha os 30 melhores matches da melhor imagem
    best_drawn_matches = best_matches[:30] if len(best_matches) >= 30 else best_matches
    best_img_matches = cv2.drawMatches(query_img, kp_query, best_image, best_kp, best_drawn_matches, None, flags=2)
    
    # Salva a imagem de melhor matching
    best_save_path = os.path.join(output_folder, "best_match_" + best_image_name)
    cv2.imwrite(best_save_path, best_img_matches)
    print("Melhor imagem de matching salva em:", best_save_path)
    
    # Exibe a melhor imagem de matching até que uma tecla seja pressionada
    #cv2.imshow("Melhor Correspondencia - " + best_image_name, best_img_matches)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
