import matplotlib.pyplot as plt

from main import load_data

src_img, dst_img, match_p_src, match_p_dst = load_data()

plt.imshow(src_img)
plt.scatter(match_p_src[0],match_p_src[1])
plt.show()

plt.imshow(dst_img)
plt.scatter(match_p_dst[0],match_p_dst[1])
plt.show()