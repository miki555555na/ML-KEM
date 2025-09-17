#ML-KEMの実装
#n=8 q=17 k=2 の実装例(簡略版)
import hashlib
import utils_mini_n8_q17
import kpke_mini_n8_q17

#import os
#import math



#パラメータ
n = 8 #多項式の次数
q = 17 #係数の法
k = 2 #多項式の数 (k=eta) 
psi = pow(3, (q-1) // n, q) #n次の原始根


#=====ハッシュ=====

def hash_H(data: bytes) -> bytes:
    #SHA3-256ハッシュ関数
    #入力：可変長のバイト列
    #出力：固定長32バイトのハッシュ値
    return hashlib.sha3_256(data).digest()

def hash_J(data: bytes, output_length: int) -> bytes:
    #SHAKE256ハッシュ関数
    #入力：可変長のバイト列
    #出力：可変長のハッシュ値
    return hashlib.shake_256(data).digest(output_length)



#ML-KEMアルゴリズム関数

#ML-KEM鍵生成

def mlkem_pke_keygen(seed1: bytes, seed2: bytes) -> tuple:
    #入力：32バイトの乱数d,z
    #出力：鍵カプセル化鍵 ek ,鍵デカプセル化鍵 dk (バイト列)

    (ek_PKE, dk_PKE) = kpke_mini_n8_q17.k_pke_keygen(seed1)
    ek = ek_PKE
    dk = dk_PKE + ek + hash_H(ek) + seed2
    return(ek,dk)

    
#ML-KEM鍵カプセル化(暗号化)

def mlkem_pke_enc(ek: bytes, message: bytes):
    #入力：鍵カプセル化鍵 ek ,乱数メッセージ message 
    #出力：共通鍵 K_final ,暗号文c

    #1.セッション鍵シード生成
    k_seed = message + hash_H(ek)

    #2.暫定鍵と乱数生成
    (K,r) = utils_mini_n8_q17.hash_G(k_seed,1)

    #3.K-PKE暗号化
    c = kpke_mini_n8_q17.k_pke_enc(ek, message, r)

    #4.共有秘密鍵生成
    K_final = hash_J(K + c,32)

    return(K_final, c, K)


#ML-KEM鍵デカプセル化(復号化)
def mlKem_pke_dec(dk: bytes,message: bytes):
    #入力：鍵デカプセル化鍵 dk ,暗号文 c'
    #出力：共通鍵K_Final

    #1.暗号文の復号

    len_dkPKE = (n * 5 * k + 7) // 8 * k  # =10 for n=8,d=5,k=2
    len_ekPKE = len_dkPKE + 32             # encoded_b (10) + p (32) = 42

    # 安全に切り出す（範囲チェックを入れる）
    if len(dk) < (len_dkPKE + 32 + 32):
        raise ValueError("dk の長さが不正です: {}".format(len(dk)))
    

    dkPKE   = dk[0:len_dkPKE]
    ekPKE   = dk[len_dkPKE:len_dkPKE + len_ekPKE]  # 実際は encoded_b(=len_dkPKE) + p(32)
    h       = dk[len_dkPKE + len_ekPKE: len_dkPKE + len_ekPKE + 32]
    z       = dk[len_dkPKE + len_ekPKE + 32: len_dkPKE + len_ekPKE + 32 + 32]

    Message = kpke_mini_n8_q17.k_pke_dec(dkPKE,message)

    #2.暫定鍵と乱数生成
    (pre_k,r) = utils_mini_n8_q17.hash_G(Message + h, 1)

    #3.暗号文の再暗号化
    C = kpke_mini_n8_q17.k_pke_enc(ekPKE, Message, r)
    #cとCを比較し、改ざん検知を行う

    #4.秘密鍵の生成
    K_Final = hash_J(pre_k + C, 32)

    return (K_Final, pre_k)




