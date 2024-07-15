
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import PyPDF2
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import csv
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')


def analisar_discurso_pdf(caminho_arquivo, num_palavras_chave=10):
    with open(caminho_arquivo, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        texto = ''
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            texto += page.extract_text()

    palavras = word_tokenize(texto.lower())
    palavras = [palavra for palavra in palavras if palavra.isalnum()]
    palavras = [palavra for palavra in palavras if palavra not in stopwords.words('portuguese')]

    contagem = Counter(palavras)
    palavras_chave = contagem.most_common(num_palavras_chave)

    sia = SentimentIntensityAnalyzer()
    sentimento = sia.polarity_scores(texto)

    return palavras_chave, sentimento


def plotar_grafico_comparativo(resultados):
    fig, ax = plt.subplots(figsize=(15, 5))

    for i, (arquivo, dados) in enumerate(resultados.items()):
        palavras_chave, _ = dados
        palavras = [palavra for palavra, frequencia in palavras_chave]
        frequencias = [frequencia for palavra, frequencia in palavras_chave]
        ax.bar(x=[x + i * 0.2 for x in range(len(palavras))], height=frequencias, width=0.2, label=arquivo)

    ax.set_xticks([x + 0.25 for x in range(len(palavras_chave))])
    ax.set_xticklabels(palavras, rotation=45, ha='right')
    ax.legend()
    plt.show()


def gerar_nuvem_palavras(resultados):
    for arquivo, dados in resultados.items():
        palavras_chave, _ = dados
        texto = ' '.join([palavra for palavra, frequencia in palavras_chave])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texto)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Nuvem de Palavras - {arquivo}')
        plt.show()


def exportar_resultados_csv(resultados, caminho_arquivo='resultados.csv'):
    with open(caminho_arquivo, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['Discurso', 'Palavra', 'Frequência', 'Sentimento Positivo', 'Sentimento Neutro', 'Sentimento Negativo',
             'Sentimento Composto'])
        for arquivo, dados in resultados.items():
            palavras_chave, sentimento = dados
            for palavra, frequencia in palavras_chave:
                writer.writerow([arquivo, palavra, frequencia, sentimento['pos'], sentimento['neu'], sentimento['neg'],
                                 sentimento['compound']])


def comparar_discursos(resultados, referencia):
    ref_palavras_chave, _ = resultados[referencia]
    ref_palavras = [palavra for palavra, _ in ref_palavras_chave]

    comparacoes = {}
    for arquivo, (palavras_chave, _) in resultados.items():
        if arquivo != referencia:
            comparacoes[arquivo] = {palavra: frequencia for palavra, frequencia in palavras_chave if
                                    palavra in ref_palavras}

    return comparacoes


caminhos_arquivos = {
    "Discurso Lula": "data/discurso1_lula.pdf",
    "Discurso Bolsonaro": "data/discurso2_bolsonaro.pdf",
    "Discurso Dilma": "data/discurso4_dilma.pdf"
}

resultados = {}
for nome_discurso, caminho_arquivo in caminhos_arquivos.items():
    resultados[nome_discurso] = analisar_discurso_pdf(caminho_arquivo)

plotar_grafico_comparativo(resultados)
gerar_nuvem_palavras(resultados)
exportar_resultados_csv(resultados)

comparacoes_lula = comparar_discursos(resultados, "Discurso Lula")
comparacoes_bolsonaro = comparar_discursos(resultados, "Discurso Bolsonaro")
comparacoes_dilma = comparar_discursos(resultados, "Discurso Dilma")

print("Comparação com Lula como referência:")
print(comparacoes_lula)

print("Comparação com Bolsonaro como referência:")
print(comparacoes_bolsonaro)

print("Comparação com Dilma como referência:")
print(comparacoes_dilma)