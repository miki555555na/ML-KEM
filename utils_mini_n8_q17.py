#共通処理(NTT,圧縮/復元,)
#n=8 q=17 k=2 の実装例(簡略版)

import hashlib

#パラメータ
n = 8 #多項式の次数
q = 17 #係数の法
k = 2 #多項式の数 (k=eta) 
psi = pow(3, (q-1) // n, q) #n次の原始根



#=====ハッシュ=====

def hash_G(data: bytes, k: int) -> tuple[bytes, bytes]:
        #SHA3-512ハッシュ関数
        #入力：可変長のバイト列、整数k
        #出力：2つの32バイトのハッシュ値
        data = data + k.to_bytes(1, 'little')
        h = hashlib.sha3_512(data).digest()
        return h[:32], h[32:]
    



