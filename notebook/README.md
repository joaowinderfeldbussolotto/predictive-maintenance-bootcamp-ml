# Notebooks - Predictive Maintenance

Esta pasta contém os Jupyter Notebooks que documentam todo o ciclo de ciência de dados do projeto, desde o entendimento dos dados até a geração dos modelos finais usados na API e na interface web.

---

## 📚 Visão Geral dos Notebooks

### 1. `projeto_final_bootcam_CD_ML_model_processing.ipynb`
- **Contextualização do Projeto:** Explica o desafio industrial, objetivos de negócio e impacto esperado.
- **Importação e Análise dos Dados:** Carregamento dos dados brutos, inspeção de tipos, valores faltantes e estatísticas descritivas.
- **Visualização:** Uso extensivo de gráficos (matplotlib, seaborn) para explorar distribuições, correlações e outliers.
- **Pré-processamento:** Imputação de valores faltantes, normalização, encoding de variáveis categóricas, criação de novas features.
- **Divisão dos Dados:** Separação em treino e teste, garantindo representatividade.
- **Exportação de Artefatos:** Geração de pipelines de preprocessamento para uso em produção.

### 2. `projeto_final_bootcam_CD_ML_binary_classification.ipynb`
- **Pipeline de Machine Learning:** Implementação completa para classificação binária de falhas (falha/não falha).
- **Testes de Algoritmos:** Avaliação de Random Forest, Gradient Boosting, Logistic Regression, XGBoost, CatBoost.
- **Balanceamento de Classes:** Uso de técnicas como SMOTE, RandomUnderSampler para lidar com desbalanceamento.
- **Validação Cruzada:** Uso de StratifiedKFold e cross_val_score para garantir robustez dos resultados.
- **Métricas:** F1-score como métrica principal, além de precisão, recall, ROC-AUC, matriz de confusão.
- **Visualização de Resultados:** Gráficos de curva ROC, curva de precisão-recall, matriz de confusão.
- **Exportação do Pipeline:** Salvamento do pipeline final (.pkl) para uso direto na API.

### 3. `projeto_final_bootcam_CD_ML_multilabel_classification.ipynb`
- **Modelagem Multi-label:** Classificação dos diferentes tipos de falha (FDF, FDC, FP, FTE, FA).
- **Estratégias de Modelagem:** Uso de MultiOutputClassifier, XGBoost, CatBoost, Random Forest, SVM, AdaBoost.
- **Validação Estratificada:** Uso de MultilabelStratifiedKFold para garantir amostragem representativa.
- **Métricas Multi-label:** Avaliação de accuracy, precision, recall, F1-score para cada classe.
- **Tuning de Hiperparâmetros:** Uso de Optuna para otimização automática dos modelos.
- **Exportação do Pipeline:** Salvamento do pipeline multi-label para futura integração na API.

---

## 🧑‍💻 Como Executar

```bash
pip install notebook pandas numpy scikit-learn xgboost catboost seaborn matplotlib optuna imblearn iterative-stratification
jupyter notebook
```
Abra os arquivos `.ipynb` no navegador para explorar, modificar e executar os experimentos.

---

## 📦 Principais Bibliotecas Utilizadas
- pandas, numpy: manipulação de dados
- scikit-learn: modelagem, métricas, pipelines
- xgboost, catboost: algoritmos avançados de ML
- matplotlib, seaborn: visualização
- imblearn: balanceamento de classes
- optuna: tuning de hiperparâmetros
- iterative-stratification: validação multi-label

---

## 🔗 Integração com o Projeto
- Os pipelines e modelos gerados são exportados para `api/ml_models/` e utilizados diretamente pela API FastAPI.
- O pipeline binário alimenta o endpoint `/predictions/binary-classification`.
- O pipeline multi-label será integrado ao endpoint `/predictions/predict` (em construção).
- Artefatos de preprocessamento podem ser usados em outros scripts ou serviços.

---

## 💡 Dicas para Uso
- Execute as células de pré-processamento antes da modelagem.
- Documente decisões e experimentos em células markdown.
- Use gráficos para justificar escolhas de features e algoritmos.
- Salve versões intermediárias dos modelos para facilitar rollback.
- Compartilhe os notebooks para facilitar revisão e colaboração.

---

## 📖 Referências
- [Documentação oficial do Jupyter](https://jupyter.org/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)
- [CatBoost Documentation](https://catboost.ai/en/docs/)
- [Optuna Documentation](https://optuna.org/)
- [Imbalanced-learn](https://imbalanced-learn.org/stable/)
- [Seaborn Examples](https://seaborn.pydata.org/examples/index.html)

---