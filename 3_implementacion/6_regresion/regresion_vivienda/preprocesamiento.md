# Metodología de preprocesamiento deducida a partir del EDA  

1. La proximidad es categorico no ordinal y por tanto debe ser codificado con metodología OneHot.(proximidad) 
2. Seleccionar los atributos mejor correlacionados.  
3. El atributo de dormitorio debe se imputado con la mediana. (SimpleImputer) 
4. Todos los atributos numéricos se va a escalar con técnica de estandarización. 
5. Detección de outliers (Con respecto al precio de vivienda). 