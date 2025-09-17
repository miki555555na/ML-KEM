#K-PKEの実装
#n=8 q=17 k=2 の実装例(簡略版)
import hashlib
import math
import utils_mini_n8_q17

#パラメータ
n = 8 #多項式の次数
q = 17 #係数の法
k = 2 #多項式の数 (k=eta) 
psi = pow(3, (q-1) // n, q) #n次の原始根



#=====擬似乱数生成関数=====

def prf(key: bytes, nonce: int, output_length: int) -> bytes:
    #SHAKE256ハッシュ関数(擬似乱数生成用=真の乱数と区別がつかないようにする)
    #nonceは乱数のような働き
    #入力：32バイトのシード、1バイトのノンス
    #出力：指定された長さの擬似乱数列
    input_data = key + nonce.to_bytes(1, 'little')
    return hashlib.shake_256(input_data).digest(output_length)


#=====XOF(SHAKE128ベース)=====

#出力の長さを指定できるハッシュ関数
class XOF:    
    #初期化
    def __init__(self):
        self.__hash__obj = hashlib.shake_128()

    #データの吸収
    def absorb(self, input_data: bytes):
        self.__hash__obj.update(input_data)
        
    #出力の絞り込み
    def squeeze(self, output_length: int) -> bytes:
        return self.__hash__obj.digest(output_length)



#=====バイト列とビット列の変換=====


#バイト列とビット列変換
def BytesToBits(byte_array: bytes) -> list[int]:
    bits = []
    for byte in byte_array:
        for i in range(8):
            bits.append((byte >> i) & 1)
    return bits


def BitsToBytes(bits: list[int]) -> bytes:
    byte_array = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            if i + j < len(bits):
                byte |= (bits[i + j] << j)
        byte_array.append(byte)
    return bytes(byte_array)

#=====ビット反転関数=====

#ビット反転
def bit_rev(x: int, bits: int) -> int:
    result = 0
    for i in range(bits):
        result = (result << 1) | (x & 1)
        x >>= 1
    return result



#=====バイト列エンコード・デコード関数=====


def ByteEncode(F: list[int], d: int) -> list[bytes]:
    #リストの各係数をdビットの2進数表記に分解して、リストの長さ(n)*dビットを作り、バイトに直す
    #入力：n個の整数リスト
    #出力：圧縮されたバイト列
    
    b = [0] * (n * d)  
    for i in range(n):
        a = F[i]
        for j in range(d):
            b[i * d + j] = a & 1
            a >>= 1
    return BitsToBytes(b)

def ByteDecode(B: bytes, d: int, k: int) -> list[list[int]]:
    #バイト列からビット列に直し、dビットの塊を2進数の数字として解釈
    #入力：圧縮されたバイト列
    #出力：k個のリスト(要素はn次元多項式)
    coeffs = []
    total_coeffs = n * k
    total_bits = len(B) * 8

    for i in range(total_coeffs):
        val = 0
        for j in range(d):
            bit_index = i * d + j
            if bit_index >= total_bits:
                break   # 範囲外に行かないようにする
            offset = bit_index // 8
            shift  = bit_index % 8
            bit = (B[offset] >> shift) & 1
            val |= bit << j
        coeffs.append(val)

    # k本のリストに分割
    return [coeffs[i*n:(i+1)*n] for i in range(k)]



#=====圧縮・復元関数=====



def Compress(vec: list[int], d: int) -> list[int]:
    #vecを0~2^d-1の範囲に圧縮する
    #入力：0~q-1までの範囲の整数のリスト
    #出力：0~2^d-1の範囲の整数のリスト
    factor = 1 << d  # 2^d
    return [((x * factor) + q ) // q for x in vec]

def Decompress(vec: list[int], d: int) -> list[int]:
    #リスト vec の各要素を Decompress して整数リストに戻す
    #d: 圧縮ビット数
    #入力：0~2^d-1の範囲の整数のリスト
    #出力：0~q-1までの範囲の整数のリスト
    factor = 1 << d  # 2^d
    return [((q * y) + (factor // 2)) // factor for y in vec]
    

    #=====多項式演算=====

def poly_add(poly1: list[int], poly2: list[int]) -> list[int]:
        #多項式加算関数
        return [(poly1[i] + poly2[i]) % q for i in range(n)]

def poly_sub(poly1: list[int], poly2: list[int]) -> list[int]:
    #多項式引算関数
    return [(poly1[i] - poly2[i] + ( q - 1 ) ) % q for i in range(n)]

def poly_mul_ntt(poly1: list[int], poly2: list[int]) -> list[int]:
    #NTT多項式の乗算関数
    return [(poly1[i] * poly2[i]) % q for i in range(n)]



#=====中心二項分布(CBD)に従う多項式の係数生成関数=====

def sample_poly_cbd(seed: bytes, eta: int) -> list[int]:
    #中心化二項分布(CBD(eta))に従って係数となる乱数を生成する
    #各係数を生成するのに、2*etaビットを使用する

    #入力：prf関数で生成した擬似乱数列(4バイト=32ビット)、
    #出力：n個の係数リスト
    
    #バイト列をビット列(32ビット)に変換
    b = BytesToBits(seed)

    coefficients = [0] * n
    for i in range (n):
        x = 0
        y = 0
        for j in range(eta):
            #2*etaビットをetaビットに分け、それぞれのビットの和(x,y)を計算
            #x-yを係数とする
            x += b[2 * i * eta + j]
            y += b[2 * i * eta + j + eta]
        coefficients[i] = (x - y) % q
    return coefficients 



#=====NTT変換=====

def sample_ntt(input_data: bytes) -> list[int]:
    #NTT多項式の係数を直接生成する関数
    #入力：32バイトのシードと行列のインデックスからなるバイト列
    #出力：NTT多項式の係数リスト(n次元)

    """
    安定版：XOF(SHAKE-128)からまとめてバイトを取り出し、
    拒否サンプリングで n 個の係数 (0..q-1) を返す。
    """

    #1.XOF初期化(SHAKE-128)
    ctx = XOF()
    #2.XOFにバイト列を吸収させる
    ctx.absorb(input_data)  
    #3.係数リストを初期化
    coefficients = []
    ## q < 2^5 (ここでは q=17)、つまり5ビット必要なので
    # 1バイトずつではなくまとめて取得して使う(毎回1バイト生成して、それを5ビットに切るのは非効率)
    # 1回に取得するバイト数（係数が足りない場合は追加で取得）
    chunk_size = 16  # 16バイトずつ取れば効率的

    buffer = b''
    while len(coefficients) < n:
        if len(buffer) < chunk_size:
            buffer += ctx.squeeze(chunk_size)

        # buffer から1バイトずつ取り出して下位5ビットを使う
        b = buffer[0]
        buffer = buffer[1:]
        val = b & 0x1F  # 下位5ビット（0..31）
        if val < q:
            coefficients.append(val)
        # もし足りなければループして追加のバイトを取りに行く

    return coefficients #n次元多項式を返す
        
def NTT(poly: list[int], psi: int) -> list[int]:
    #NTT変換関数(多項式からNTT多項式へ変換)
    #入力：n個の整数リスト(多項式),n次の原始根
    #出力：n個の整数リスト(NTT多項式)

    #係数リストのコピー(NTT変換前)
    coefficients = poly.copy()

    #回転子リストの生成
    omegas = [0] * n
    zeta = 1
    for i in range(n):
        omegas[i] = zeta
        zeta = (zeta * psi) % q

    #バタフライ演算
    #n=8のとき、length=4,2の順に変化(分けて処理し、高速化を図る)
    length = n // 2
    i = 1
    while length >= 1:
        start = 0
        while start < n:
            #回転子(昇順のべき乗に直す)
            bitrev_i = bit_rev(i, int(math.log2(n))) #n=8は3ビットで表現できる
            omega = omegas[bitrev_i]
            for j in range(start, start + length):
                t = (omega * coefficients[j + length]) % q 
                coefficients[j + length] = (coefficients[j] - t + q) % q
                coefficients[j] = (coefficients[j] + t ) % q
            start += 2 * length
            i += 1
        length //= 2
    return coefficients


def NTT_rev(poly: list[int],psi: int) -> list[int]:
    #逆NTT変換関数(NTT多項式から多項式へ変換)
    #入力：n次元のNTT多項式の係数リスト
    #出力：n次元の多項式の係数リスト

    #係数リストのコピー(逆NTT変換前)
    coefficients = poly.copy()

    #NTTとは異なる回転子を生成 (ζ^-1)
    psi_rev = pow(psi, -1, q)

    #回転子リストの生成
    omegas = [0] * n
    zeta = 1

    for i in range(n):
        omegas[i] = zeta
        zeta = (zeta * psi_rev) % q

    #バタフライ演算
    length = n // 2
    i = n // 2 - 1

    while length >= 1:
        start = 0
        while start < n:
            #回転子
            bitrev_i = bit_rev(i, int(math.log2(n))) #n=8は3ビットで表現できる
            omega = omegas[bitrev_i]
            for j in range(start, start + length):
                t = coefficients[j] % q
                coefficients[j] = (t + coefficients[j + length] ) % q
                coefficients[j + length] = (omega * (coefficients[j + length] - t) + q) % q
            start += 2 * length
            i -= 1
        length //= 2
    return coefficients




#=====K-PKEアルゴリズム関数======


#K-PKE鍵生成
def k_pke_keygen(seed: bytes) -> tuple:
    #入力：32バイトの乱数
    #出力：公開鍵(ek_PKE), 秘密鍵(dk_PKE)

    #1.乱数生成[p,S]
    p,S = utils_mini_n8_q17.hash_G(seed, k)

    #2.公開鍵行列ntt_A(k x k)の生成<mod q>
    #乱数pを使用してn=8次のNTT多項式生成(k×k)
    ntt_A = [[0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            input_data = p + j.to_bytes(1, 'little') + i.to_bytes(1, 'little')
            ntt_A[i][j] =sample_ntt(input_data)
        
    #3-1.秘密鍵ベクトルs(k)の生成<mod q>
    #乱数Sを使用してn=8次の多項式生成(k)
    s = [0] * k
    nonce_s = 0
    for i in range(k):
        s[i] = sample_poly_cbd(prf(S, nonce_s, 4), 2)
        nonce_s += 1

    #3-2.誤差多項式e(k)の生成<mod q>
    #乱数Sを使用してn=8次の多項式生成(k)
    e = [0] * k
    nonce_e = k
    for i in range(k):
        e[i] = sample_poly_cbd(prf(S, nonce_e, 4),2)
        nonce_e += 1 

    #4.秘密鍵sと誤差多項式eをNTT変換<mod q>
    ntt_vec_s = [NTT(poly, psi) for poly in s]
    ntt_vec_e = [NTT(poly, psi) for poly in e]

    #5.公開鍵b(k)の生成<mod q>
    #b = ntt_A * ntt_vec_s + ntt_vec_e
    #n=8次の多項式生成(k)
    b = [0] * k
    for i in range(k):
        term_ntt = [0] * n
        for j in range(k):
            prod = poly_mul_ntt(ntt_A[i][j], ntt_vec_s[j])
            term_ntt = poly_add(term_ntt, prod)
        b[i] = poly_add(term_ntt, ntt_vec_e[i])

    #6-1.公開鍵bのエンコード
    #精度を維持しながらエンコード(圧縮)する
    #多項式の係数を5ビットの2進数表現にし、バイト列にエンコードする
    #k個のリスト(要素はn*5/8=5バイト)=全体としては10バイト
    encoded_b = []
    for i in range(k):
        encoded_part = ByteEncode(b[i], 5)
        encoded_b.append(encoded_part)
    
    #6-2.秘密鍵sのエンコード
    encoded_s = []
    for i in range(k):
        encoded_part = ByteEncode(ntt_vec_s[i], 5)
        encoded_s.append(encoded_part)

    #7-1.公開鍵バイト列を結合
    #10バイト+32バイト=42バイト
    ek_PKE = b''.join(encoded_b) + p 

    #7-2秘密鍵バイト列を結合
    dk_PKE = b''.join(encoded_s)    

    return (ek_PKE, dk_PKE)

#K-PKE鍵カプセル化(暗号化)
def k_pke_enc(ek_PKE: bytes,m: bytes, r: bytes) -> bytes:
    #入力：公開鍵、メッセージ(1バイト)、乱数
    #出力：暗号文

    #1.公開鍵のデコード(bとpを復元)
    p = ek_PKE[-32:]#最後の32バイト
    encorded_b = ek_PKE[:-32]#最初の部分
    b = ByteDecode(encorded_b, 5, k)#n次多項式(k)(2個のリスト(要素8個の整数))


    #2.pを使用して、ntt_A(k x k)を再現
    #n=8次の多項式生成(k x k)
    ntt_A = [[0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            input_data = p + j.to_bytes(1, 'little') + i.to_bytes(1, 'little')
            ntt_A[i][j] = sample_ntt(input_data)
    
    #3-1.一時乱数ベクトルy(k)の生成
    #n=8次の多項式生成(k)
    nonce = 0
    y = [0] * k
    for i in range(k):
        y[i] = sample_poly_cbd(prf(r, nonce, 4), 2)
        nonce +=1
    
    #3-2.誤差ベクトルe1(k)の生成
    #n=8次の多項式生成(k)
    e1 = [0] * k
    for i in range(k):
        e1[i] = sample_poly_cbd(prf(r, nonce, 4), 2)
        nonce += 1
    
    #4.誤差多項式e2(1本のn次元多項式)の生成
    e2 = sample_poly_cbd(prf(r, nonce, 4), 2)

    #5.一時乱数ベクトルをNTT変換(k)
    #n=8次のNTT多項式生成(k)
    ntt_y = [NTT(poly,psi) for poly in y]#(2個のリスト(要素8個の整数))

    #6.ベクトルUの生成(U = ntt_A * ntt_y + e1)
    #k個のリスト(要素はn次多項式)
    psi_rev = pow(psi, -1, q)  # 逆元 (NTT の逆回転子)
    U = [] 
    for i in range(k):
        term_ntt = [0] * n
        for j in range(k):
            prod = poly_mul_ntt(ntt_A[j][i], ntt_y[j])
            term_ntt = poly_add(term_ntt, prod)
        U.append(poly_add(NTT_rev(term_ntt, psi_rev), e1[i])) #逆NTT
    
    #7.メッセージm(1バイト)の多項式化
    #デコードして多項式の係数の範囲を{0,1}に(ByteDecode)
    #多項式の係数を{0,1}から0~q-1に(Decompress)
    #1本のn次多項式
    μ_poly_list = ByteDecode(m, 1, 1)  # 返り値は[[1,0,1,1,...]] のような形式
    μ = Decompress(μ_poly_list[0], 1)# μ_poly_list[0] を使う
    
    #8.メッセージの暗号化
    #v = b * ntt_y + e2 + μ
    #1本のn次元多項式
    # 1本のn次多項式を作る
    term_ntt = [0] * n
    for i in range(k):  # k = 2
        prod = poly_mul_ntt(b[i], ntt_y[i])  # b[i], ntt_y[i] は長さ 8 のリスト
        term_ntt = poly_add(term_ntt, prod)

    # e2 + μ を足して最終的な V
    V = poly_add(term_ntt, poly_add(e2, μ))  # 長さ8のリスト

    #9.u(k)とv(1)を圧縮
    #uは、k個のリスト(要素はn次多項式)
    #vは、1本のn次元多項式
    #n次多項式の係数を{0,1}で表し、バイトにする
    u = [ByteEncode(Compress([coeff for coeff in U[i]], 1), 1) for i in range(k)]#nxk/8バイト
    v = ByteEncode(Compress(V,1), 1)#n/8バイト
    c = b''.join(u) + v#2+1=3バイト

    return c


#K-PKE鍵デカプセル化(復号化)
def k_pke_dec(dk_PKE: list[int], c: bytes) -> bytes:
    #入力：秘密鍵 dk_PKE(バイト列)、暗号文 c(u,v)(バイト列)
    #出力：復号文m'

    #1.暗号文のデコード(u,vを復元)
    u = c[0:2]
    v = c[2:]

    #2.UとVを復元
    #n=8次のNTT多項式生成(k)
    #バイト列(2バイト)からビット列(16ビット)に直し、k個のリスト(要素はn個のリスト)にする(ByteDecode)
    #k個のリストの要素である多項式の係数を0~q-1にする(Decompress)
    U = [Decompress(i, 1) for i in ByteDecode(u, 1, k)]
    #1本のn次多項式
    #バイト列(1バイト)からビット列(8ビット)に直し、一個のリスト(要素はn個のリスト)にする(ByteDecode)
    #リストの要素である多項式の係数を0~q-1に(Decompress)
    V = [Decompress(i, 1) for i in ByteDecode(v, 1, 1)]

    #3.秘密鍵(encorded_s)のデコード
    #n=8次のNTT多項式生成(k)
    #バイト列をk個の分け、ビットに直し、5ビットの塊を一つの係数として扱う
    s = ByteDecode(dk_PKE, 5, k)  # k本のn次多項式

    #4.中間多項式w を生成
    #w = V - s * NTT(U)
    #1本のn次多項式
    ntt_U = [NTT(poly,psi) for poly in U] #k個のNTT多項式
    psi_rev = pow(psi, -1, q)  # 逆元 (NTT の逆回転子)
    term_ntt = [0] * n
    for j in range(k):
        prod = poly_mul_ntt(s[j], ntt_U[j])  
        term_ntt = poly_add(term_ntt, prod)
    w = poly_sub(V[0],NTT_rev(term_ntt,psi_rev))

    #5.メッセージm'の復元
    #暗号化で行った処理の逆を実行
    #係数を{0,1}に丸め、バイト列に変換
    m = ByteEncode(Compress(w,1), 1)

    return m