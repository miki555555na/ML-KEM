#実行テスト
#n=8 q=17 k=2 の実装例(簡略版)
import os
import mlkem_mini_n8_q17

#パラメータ
n = 8 #多項式の次数
q = 17 #係数の法
k = 2 #多項式の数 (k=eta) 
psi = pow(3, (q-1) // n, q) #n次の原始根


#テスト関数

def approx_equal(a: bytes, b: bytes, tolerance: int = 7) -> bool:
    """
    バイト列aとbが誤差tolerance以内で一致するか確認する
    """
    if len(a) != len(b):
        return False
    return all(abs(x - y) <= tolerance for x, y in zip(a, b))


def main():

    seed1 = os.urandom(32)
    seed2 = os.urandom(32)

    # 鍵生成
    ek, dk = mlkem_mini_n8_q17.mlkem_pke_keygen(seed1,seed2)
    print("[*] 鍵カプセル化鍵:", ek.hex())
    print("[*] 鍵デカプセル鍵:", dk.hex())

    # テスト用メッセージと乱数
    m = b"B"   # 1バイトメッセージ
    #r = os.urandom(32)


    # カプセル化
    K_final, c, K = mlkem_mini_n8_q17.mlkem_pke_enc(ek, m)
    print("[*] 暗号文:", c.hex())
    print("[*] カプセル化側の共通鍵:", K.hex())

    k_final, pre_k = mlkem_mini_n8_q17.mlKem_pke_dec(dk,c)



    print("[*] デカプセル化側の共通鍵:",pre_k.hex())


    if K_final == K_final:
        print("復号成功: 最終共通鍵が完全一致")
    else:
        print("復号失敗: 最終共通鍵が一致しない")
    



if __name__ == "__main__":
    main()
