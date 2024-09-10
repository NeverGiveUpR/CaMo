import numpy as np

def nguyen1(n=100):
    # n = 100
    range_min, range_max = [-1, 1]
    # x = np.linspace(-1, 1, n)
    # x = np.random.randn(n)
    x = (range_max - range_min) * np.random.rand(n) + range_min
    # y = x*3 + x*2 + x
    y = x**3 + x**2 + x
    return x[:, np.newaxis], y

def nguyen2(n = 100):
    # x = np.linspace(-1, 1, n)
    range_min, range_max = [-1, 1]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    # y = x*4 + x*3 + x*2 + x
    y = x**4 + x**3 + x**2 + x
    return x[:, np.newaxis], y

def nguyen3(n = 100):
    # x = np.linspace(-1, 1, n)
    range_min, range_max = [-1, 1]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    # y = x*5 + x*4 + x*3 + x*2 + x
    y = x**5 + x**4 + x**3 + x**2 + x
    return x[:, np.newaxis], y

def nguyen4(n = 100):
    range_min, range_max = [-1, 1]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    # y = x*6 + x*5 + x*4 + x*3 + x*2 + x
    y = x**6 + x**5 + x**4 + x**3 + x**2 + x
    return x[:, np.newaxis], y

def nguyen5(n = 100):
    range_min, range_max = [-1, 1]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    y = np.sin(x**2)*np.cos(x) - 1
    return x[:, np.newaxis], y

def nguyen6(n = 100):
    # x = np.linspace(-1, 1, n)
    range_min, range_max = [-1, 1]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    y = np.sin(x) + np.sin(x+x**2)
    return x[:, np.newaxis], y

def nguyen7(n = 100):
    # x = np.linspace(0, 2, n)
    range_min, range_max = [-1, 1]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    y = np.log(x+1)+np.log(x**2+1)
    return x[:, np.newaxis], y

def nguyen8(n = 100):
    # x = np.linspace(0, 4, n)
    range_min, range_max = [0, 4]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    y = np.sqrt(x)
    return x[:, np.newaxis], y

def nguyen9(n = 100):
    # x = np.linspace(0, 1, n)
    # np.random.shuffle(x)
    # y = np.linspace(0, 1, n)
    # np.random.shuffle(y)
    range_min, range_max = [-1, 1]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    y = (range_max - range_min) * np.random.rand(n) + range_min
    X = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
    f = np.sin(x) + np.sin(y**2)
    return X, f

def nguyen10(n = 100):
    range_min, range_max = [-1, 1]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    y = (range_max - range_min) * np.random.rand(n) + range_min
    X = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
    f = 2*np.sin(x)*np.cos(y)
    return X, f

def nguyen11(n = 100):
    range_min, range_max = [0, 1]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    y = (range_max - range_min) * np.random.rand(n) + range_min
    X = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
    f = x**y
    return X, f

def nguyen12(n = 100):
    range_min, range_max = [-1, 1]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    y = (range_max - range_min) * np.random.rand(n) + range_min
    X = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
    f = x**4 - x**3 + 0.5*y**2 - y
    return X, f

def nguyen2_(n = 100):
    range_min, range_max = [-1, 1]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    # y = 4*x*4 + 3*x*3 + 2*x*2 + x
    y = 4*x**4 + 3*x**3 + 2*x**2 + x
    return x[:, np.newaxis], y

def nguyen5_(n = 100):
    range_min, range_max = [-1, 1]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    y = np.sin(x**2)*np.cos(x) - 2
    return x[:, np.newaxis], y

def nguyen8_(n = 100):
    range_min, range_max = [0, 4]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    y = x**(1/3)
    return x[:, np.newaxis], y

def nguyen8_2(n = 100):
    range_min, range_max = [0, 4]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    y = x**(2/3)
    return x[:, np.newaxis], y

def nguyen1_c(n = 100):
    range_min, range_max = [-1, 1]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    y = 3.39*x**3 + 2.12*x**2 + 1.78*x
    return x[:, np.newaxis], y

def nguyen5_c(n = 100):
    range_min, range_max = [-1, 1]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    y = np.sin(x**2)*np.cos(x) - 0.75
    return x[:, np.newaxis], y

def nguyen7_c(n = 100):
    range_min, range_max = [-1, 1]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    y = np.log(x+1.4)+np.log(x**2+1.3)
    return x[:, np.newaxis], y

def nguyen8_c(n = 100):
    range_min, range_max = [0, 4]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    y = np.sqrt(1.23*x)
    return x[:, np.newaxis], y

def nguyen10_c(n = 100):
    range_min, range_max = [-1, 1]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    y = (range_max - range_min) * np.random.rand(n) + range_min
    X = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
    f = np.sin(1.5*x)*np.cos(0.5*y)
    return X, f

def grammarVAE1(n = 1000):
    range_min, range_max = [-1, 1]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    y = 1/3 + x + np.sin(x)
    return x[:, np.newaxis], y

def Jin1(n = 1000):
    range_min, range_max = [-3, 3]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    y = (range_max - range_min) * np.random.rand(n) + range_min
    X = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
    f = 2.5*x**4-1.3*x**3+0.5*y**2-1.7*y
    return X, f

def Jin2(n = 1000):
    range_min, range_max = [-3, 3]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    y = (range_max - range_min) * np.random.rand(n) + range_min
    X = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
    f = 8.0*x**2+8.0*y**3-15.0
    return X, f

def Jin3(n = 1000):
    range_min, range_max = [-3, 3]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    y = (range_max - range_min) * np.random.rand(n) + range_min
    X = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
    f = 0.2*x**2+0.5*y**3-1.2*y-0.5*x
    return X, f

def Jin4(n = 1000):
    range_min, range_max = [-3, 3]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    y = (range_max - range_min) * np.random.rand(n) + range_min
    X = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
    f = 1.5*np.exp(x) + 5.0*np.cos(y)
    return X, f

def Jin5(n = 1000):
    range_min, range_max = [-3, 3]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    y = (range_max - range_min) * np.random.rand(n) + range_min
    X = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
    f = 6*np.sin(x)*np.cos(y)
    return X, f

def Jin6(n = 1000):
    range_min, range_max = [-3, 3]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    y = (range_max - range_min) * np.random.rand(n) + range_min
    X = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
    f = 1.35*x*y + 5.5*np.sin((x-1.0)*(y-1.0))
    return X, f

def Neat8(n = 1000):
    range_min, range_max = [0.3, 4]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    y = (range_max - range_min) * np.random.rand(n) + range_min
    X = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
    f = np.exp(-(x-1)**2)/(1.2+(y-2.5)**2)
    return X, f

def Neat9(n = 1000):
    range_min, range_max = [-5, 5]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    y = (range_max - range_min) * np.random.rand(n) + range_min
    X = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
    f = 1/(1+x**(-4)) + 1/(1+y**(-4))
    return X, f

def R1(n = 100):
    range_min, range_max = [-2, 2]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    f = (x+1)**3/(x**2-x+1)
    return x[:, np.newaxis], f

def R2(n = 100):
    range_min, range_max = [-2, 2]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    f = (x**5-3*x**3+1)/(x**2+1)
    return x[:, np.newaxis], f

def R3(n = 100):
    range_min, range_max = [-1, 0.5]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    f = (x**6+x**5)/(x**4+x**3+x**2+x+1)
    return x[:, np.newaxis], f

def Livermore4(n = 100):
    range_min, range_max = [0.1, 2.2]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    f = np.log(x+1)+np.log(x**2+1)+np.log(x)
    return x[:, np.newaxis], f

def Livermore5(n = 100):
    range_min, range_max = [0, 1.2]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    range_min, range_max = [-1, 1]
    y = (range_max - range_min) * np.random.rand(n) + range_min
    X = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
    f = x**4-x**3+x**2-y
    return X, f

def Livermore9(n = 100):
    range_min, range_max = [-1.2, 0.6]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    f = x**9+x**8+x**7+x**6+x**5+x**4+x**3+x**2+x
    return x[:, np.newaxis], f

def Livermore11(n = 1000):
    range_min, range_max = [-1, 1]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    range_min, range_max = [-10, 10]
    y = (range_max - range_min) * np.random.rand(n) + range_min
    X = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
    f = x**2*x**3/(x+y)
    return X, f

def Livermore12(n = 1000):
    range_min, range_max = [-2, -1]
    x_neg = (range_max - range_min) * np.random.rand(int(n/2)) + range_min
    range_min, range_max = [-1, 2]
    x_pos = (range_max - range_min) * np.random.rand(int(n/2)) + range_min
    x = np.concatenate([x_neg, x_pos])
    np.random.shuffle(x)
    range_min, range_max = [-2, -1]
    y_neg = (range_max - range_min) * np.random.rand(int(n/2)) + range_min
    range_min, range_max = [-1, 2]
    y_pos = (range_max - range_min) * np.random.rand(int(n/2)) + range_min
    y = np.concatenate([y_neg, y_pos])
    np.random.shuffle(y)
    X = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
    f = x**5/y**3
    return X, f

def Livermore14(n = 1000):
    range_min, range_max = [-4.4, 4.4]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    f = x**3+x**2+x+np.sin(x)+np.sin(x**2)
    return x[:, np.newaxis], f

def Livermore15(n = 100):
    range_min, range_max = [0, 1]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    f = x**(1/5)
    return x[:, np.newaxis], f

def Livermore16(n = 100):
    range_min, range_max = [0, 1]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    f = x**(2/5)
    return x[:, np.newaxis], f

def Livermore17(n = 1000):
    range_min, range_max = [-5, 5]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    y = (range_max - range_min) * np.random.rand(n) + range_min
    X = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
    f = 4*np.sin(x)*np.cos(y)
    return X, f

def Livermore18(n = 100):
    range_min, range_max = [-2, 2]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    f = np.sin(x**2)*np.cos(x)-5
    return x[:, np.newaxis], f

def Livermore19(n = 100):
    range_min, range_max = [-1, 1]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    f = x**5+x**4+x**2+x
    return x[:, np.newaxis], f

def Livermore20(n = 100):
    range_min, range_max = [-2, 2]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    f = np.exp(-x**2)
    return x[:, np.newaxis], f

def Livermore21(n = 100):
    range_min, range_max = [-1.2, 0.7]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    f = x**8+x**7+x**6+x**5+x**4+x**3+x**2+x
    return x[:, np.newaxis], f

def Livermore22(n = 100):
    range_min, range_max = [-4, 4]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    f = np.exp(-0.5*x**2)
    return x[:, np.newaxis], f

def Koza2(n = 100):
    range_min, range_max = [-2, 2]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    f = x**5-2*x**3+x
    return x[:, np.newaxis], f

def Koza3(n = 100):
    range_min, range_max = [-1.2, 1.2]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    f = x**6-2*x**4+x**2
    return x[:, np.newaxis], f

def Keijzer3(n = 100):
    range_min, range_max = [-1, 1]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    f = 0.3*x*np.sin(2*np.pi*x)
    return x[:, np.newaxis], f

def Keijzer4(n = 100):
    range_min, range_max = [-2, 5]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    f = x**3*np.exp(-x)*np.cos(x)*np.sin(x)*(np.sin(x**2)*np.cos(x)-1)
    return x[:, np.newaxis], f

def Keijzer6(n = 100):
    range_min, range_max = [-2, 1]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    f = x*(x+1)/2
    return x[:, np.newaxis], f

def Keijzer7(n = 100):
    range_min, range_max = [0.1, 4]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    f = np.log(x)
    return x[:, np.newaxis], f

def Keijzer9(n = 100):
    range_min, range_max = [-6, 6]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    f = np.log(x+np.sqrt(x**2+1))
    return x[:, np.newaxis], f

def Keijzer11(n = 1000):
    range_min, range_max = [-3, 3]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    y = (range_max - range_min) * np.random.rand(n) + range_min
    f = x*y+np.sin((x-1)*(y-1))
    X = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
    return X, f

def Keijzer14(n = 1000):
    range_min, range_max = [-0.5, 0.5]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    range_min, range_max = [3, 4]
    y = (range_max - range_min) * np.random.rand(n) + range_min
    f = 8/(2+x**2+y**2)
    X = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
    return X, f

def Keijzer15(n = 1000):
    range_min, range_max = [-2, 2]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    range_min, range_max = [0, 2]
    y = (range_max - range_min) * np.random.rand(n) + range_min
    f = x**3/5+y**2/2-y-x
    X = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
    return X, f

def Constant4(n = 1000):
    range_min, range_max = [0, 2]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    range_min, range_max = [-2, 2]
    y = (range_max - range_min) * np.random.rand(n) + range_min
    f = 2.7*x**y
    X = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
    return X, f

def Constant6(n = 100):
    range_min, range_max = [0.1, 5]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    f = x**(0.426)
    return x[:, np.newaxis], f

def Constant7(n = 1000):
    range_min, range_max = [-5, 5]
    x = (range_max - range_min) * np.random.rand(n) + range_min
    range_min, range_max = [-2, 2]
    y = (range_max - range_min) * np.random.rand(n) + range_min
    f = 2*np.sin(1.3*x)*np.cos(y)
    X = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
    return X, f

func_dict = {'nguyen1': nguyen1, 'nguyen2': nguyen2, 'nguyen3': nguyen3, 'nguyen4': nguyen4,
                'nguyen5': nguyen5, 'nguyen6': nguyen6, 'nguyen7': nguyen7, 'nguyen8': nguyen8,
                'nguyen9': nguyen9, 'nguyen10': nguyen10, 'nguyen11': nguyen11, 'nguyen12': nguyen12,
                'nguyen2_': nguyen2_, 'nguyen5_': nguyen5_, 'nguyen8_': nguyen8_, 'nguyen8_2':nguyen8_2,
                'nguyen1_c': nguyen1_c, 'nguyen5_c': nguyen5_c, 'nguyen7_c': nguyen7_c, 'nguyen8_c': nguyen8_c,
                'nguyen10_c': nguyen10_c,
                'grammarVAE1': grammarVAE1,
                'Jin1': Jin1, 'Jin2': Jin2, 'Jin3': Jin3, 'Jin4': Jin4, 'Jin5': Jin5, 'Jin6': Jin6,
                'Neat8': Neat8, 'Neat9': Neat9,
                'R1': R1, 'R2': R2, 'R3': R3, 'Livermore4': Livermore4, 'Livermore5':Livermore5,
                   'Livermore9': Livermore9, 'Livermore11': Livermore11, 'Livermore12':Livermore12,
                   'Livermore14':Livermore14, 'Livermore15': Livermore15, 'Livermore16':Livermore16,
                   'Livermore17':Livermore17, 'Livermore18':Livermore18, 'Livermore19':Livermore19,
                   'Livermore20':Livermore20, 'Livermore21':Livermore21, 'Livermore22':Livermore22,
                   'Koza2':Koza2, 'Koza3':Koza3, 'Keijzer3':Keijzer3,'Keijzer4':Keijzer4,
                   'Keijzer6':Keijzer6, 'Keijzer7':Keijzer7, 'Keijzer9':Keijzer9, 'Keijzer11':Keijzer11,
                   'Keijzer14':Keijzer14, 'Keijzer15':Keijzer15, 'Constant4':Constant4, 'Constant6': Constant6,
                   'Constant7':Constant7
}

def generate_even_parity(n=100, d=2):
    # d is the number of digits, e.g., 01, 11, 1
    # parity check, odd check
    x = np.random.randint(2, size=(n, d))
    y = [True if np.sum(xi)%2!=0 else False for xi in x]
    return np.array(x), np.array(y)
