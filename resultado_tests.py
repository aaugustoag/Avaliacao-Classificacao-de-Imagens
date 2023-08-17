from sklearn.exceptions import UndefinedMetricWarning
from resultado import Resultado,Fold
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import unittest
import warnings
class TestResultado(unittest.TestCase):
    y =         np.array([0,0,1,1,1,2,2,2,2,2,2,2,2])
    predict_y = np.array([0,1,1,2,2,1,2,1,2,0,2,2,1])
    y_zero = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])
    predict_y_zero = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])
    def matriz_teste(self,resultado,matriz_esperada):
        for classe_real in range(len(matriz_esperada)):
            for classe_prevista in range(len(matriz_esperada)):
                qtd_esperada = matriz_esperada[classe_real][classe_prevista]
                qtd = resultado.mat_confusao[classe_real][classe_prevista]
                self.assertEqual(qtd_esperada,
                                qtd,
                                "Deveriam ter {qtd_esperada} elementos da classe {classe_real}"\
                                " previstos como classe {classe_prevista} mas existiam {qtd_atual}"\
                                .format(qtd_esperada=qtd_esperada,qtd_atual=qtd,classe_prevista=classe_prevista,
                                        classe_real=classe_real))
    def test_mat_confusao(self):

        #teste simples
        resultado = Resultado(TestResultado.y,TestResultado.predict_y)
        matriz_esperada = [[1,1,0],
                            [0,1,2],
                            [1,3,4]]
        self.matriz_teste(resultado,matriz_esperada)

        #teste quando classe real = 0
        resultado = Resultado(TestResultado.y_zero,TestResultado.predict_y)
        matriz_esperada = [[2,5,6],
                            [0,0,0],
                            [0,0,0]]
        self.matriz_teste(resultado,matriz_esperada)

        #teste quando classe predita = 0
        resultado = Resultado(TestResultado.y,TestResultado.predict_y_zero)
        matriz_esperada = [[2,0,0],
                            [3,0,0],
                            [8,0,0]]
        self.matriz_teste(resultado,matriz_esperada)



    def metric_test(self,nom_metrica,expected,current):
        for classe,exp_val in enumerate(expected):
            self.assertAlmostEqual(current[classe],exp_val,
                    msg="A {nom_metrica} da classe {classe} deveria ser {prec_esperada} mas é {prec}"
                    .format(nom_metrica=nom_metrica,classe=classe,prec_esperada=exp_val,prec=current[classe]))

    def test_acuracia(self):
        resultado = Resultado(TestResultado.y,TestResultado.predict_y)
        self.assertAlmostEqual(resultado.micro_f1,6/13,msg="Acurácia não está com o valor esperado")

    def test_precisao(self):
        resultado = Resultado(TestResultado.y,TestResultado.predict_y)
        precisao_esperada = [1/2,1/5,4/6]
        self.metric_test('precisao',precisao_esperada,resultado.precisao)

        #testa quando há apenas classe 0
        resultado = Resultado(TestResultado.y_zero,TestResultado.predict_y)
        precisao_esperada = [2/2,0,0]
        self.metric_test('precisao',precisao_esperada,resultado.precisao)

        #testa quando foi previsto apenas 0
        resultado = Resultado(TestResultado.y,TestResultado.predict_y_zero)
        precisao_esperada = [2/13,0,0]
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            self.metric_test('precisao',precisao_esperada,resultado.precisao)
            # Verify some things
            self.assertTrue(len(w) >= 1,"Não foi tratado o warning conforme especificado")
            self.assertTrue(issubclass(w[-1].category, UndefinedMetricWarning),"O warning deveria ser da classe UndefinedMetricWarning do scikit learn")


    def test_revocacao(self):
        resultado = Resultado(TestResultado.y,TestResultado.predict_y)
        revocacao_esperada = [1/2,1/3,4/8]
        self.metric_test('revocação',revocacao_esperada,resultado.revocacao)

        #testa quando ha apenas a classe 0
        resultado = Resultado(TestResultado.y_zero,TestResultado.predict_y)
        revocacao_esperada = [2/13, 0, 0]
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            self.metric_test('revocação',revocacao_esperada,resultado.revocacao)
            # Verify some things
            self.assertTrue(len(w) >= 1,"Não foi tratado o warning conforme especificado")
            self.assertTrue(issubclass(w[-1].category, UndefinedMetricWarning),"O warning deveria ser da classe UndefinedMetricWarning do scikit learn")


        #testa quando Foi previsto apenas a classe zero
        resultado = Resultado(TestResultado.y,TestResultado.predict_y_zero)
        revocacao_esperada = [2/2,0,0]
        self.metric_test('revocação',revocacao_esperada,resultado.revocacao)

    def test_f1_por_classe(self):
        resultado = Resultado(TestResultado.y,TestResultado.predict_y)
        prec = [1/2,1/5,4/6]
        rev = [1/2,1/3,4/8]
        f1_esp = [2*(prec[i]*rev[i])/(prec[i]+rev[i]) for i in range(len(prec))]

        self.metric_test('f1',f1_esp,resultado.f1_por_classe)

    def test_macro_f1(self):
        resultado = Resultado(TestResultado.y,TestResultado.predict_y)
        prec = [1/2,1/5,4/6]
        rev = [1/2,1/3,4/8]
        f1_esp = [2*(prec[i]*rev[i])/(prec[i]+rev[i]) for i in range(len(prec))]

        macro_f1 = np.average(f1_esp)
        self.assertAlmostEqual(resultado.macro_f1,macro_f1,msg="Macro F1 não está com o valor esperado")


class TestFold(unittest.TestCase):
    df_treino = pd.DataFrame({"A":[1, 1, 2, 2, 3, 4, 4, 5, 6, 1],
                    "B": [True,False,True,False,True,False,False,False,False,True],
                    "C":[23, 3, 123, 55, 12,33,44,21,55,22],
                    "D":[1, 1,  1, 1, 1, 1 , 1 , 1 , 1 , 1 ],
                    "realClass":[1,1,0,0,0,1,1,0,1,0]})

    df_teste = pd.DataFrame({"A":[1,1,1,2,3,3,3,3,4,4,4,4,5,5,5],
                    "B": [True,False,True,True,False,True,True,False,True,True,False,True,True,False,True],
                    "C":[333,-1,5,333,-12,52,3323,-12,52,3323,-41,53,3333,-12,51],
                    "D":[2, 2, 3,2, 2, 3,2, 23, 3,2, 21, 3,2, 22, 3],
                    "realClass":[1,0,1,1,0,1,1,0,1,1,0,0,0,0,1]})
    df_dados = pd.DataFrame({"A":[1, 1, 2, 2, 3, 4, 4, 5, 6,1,1,1,1,2,3,3,3,3,4,4,4,4,5,5,5],
                    "B": [True,False,True,False,True,False,False,False,False,True,
                         True,False,True,True,False,True,True,False,True,True,False,True,True,False,True],
                    "C":[23, 3, 123, 55, 12,33,44,21,55,22,333,-1,5,333,-12,52,3323,-12,52,3323,-41,53,3333,-12,51],
                    "D":[1,  1, 1, 1, 1, 1, 1, 1, 1,1,2, 2, 3,2, 2, 3,2, 23, 3,2, 21, 3,2, 22, 3],
                    "realClass":[1,1,1,1,2,2,2,2,2,2,2,2,2,2,0,1,1,0,1,1,0,0,0,0,1]})
    def test_eval(self):
        clf_dtree = DecisionTreeClassifier(random_state=1)
        fold = Fold(TestFold.df_treino,TestFold.df_teste,"realClass")
        resultado = fold.eval(clf_dtree)
        acuracia = resultado.acuracia
        macro_f1 = resultado.macro_f1
        print("Macro f1: {macro} Acuracia: {acuracia}".format(macro=macro_f1,acuracia=acuracia))

        self.assertAlmostEqual(macro_f1, 0.5982142857142857,msg="Macro F1 não está com o valor esperado")
        self.assertAlmostEqual(acuracia, 0.6,msg="Acuracia não está com o valor esperado")

if __name__ == "__main__":
    unittest.main()
