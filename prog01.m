% Cargar los datos del archivo CSV
trainData = readtable('train.csv');
testData = readtable('test.csv');

% Preprocesamiento de los datos de entrenamiento
medianaEdad = median(trainData.Age, 'omitnan');
trainData.Age(isnan(trainData.Age)) = medianaEdad;

% Preprocesamiento de los datos de prueba
testData.Age(isnan(testData.Age)) = medianaEdad;  % Rellenar los valores faltantes de Age en testData con la mediana

% Convertir 'Sex' a una variable binaria en ambos conjuntos
trainData.Sex = strcmp(trainData.Sex, 'female');
testData.Sex = strcmp(testData.Sex, 'female');

% Seleccionar las características relevantes de ambos conjuntos
XTrain = [trainData.Pclass, trainData.Sex, trainData.Age, trainData.Fare];
yTrain = trainData.Survived + 1;  % Convertir Survived a 1/2 en lugar de 0/1

XTest = [testData.Pclass, testData.Sex, testData.Age, testData.Fare];

% Entrenar el modelo de regresión logística
B = mnrfit(XTrain, yTrain, 'model', 'nominal');

% Realizar predicciones en el conjunto de prueba
p = mnrval(B, XTest, 'model', 'nominal');
yPred = p(:,2) >= 0.5;  % Esto generaría las predicciones

% Calcular métricas y resultados adicionales
edadPromedio = mean(XTest(:, 3));
numPrimeraClase = sum(XTest(:, 1) == 1);
porcentajeSupervivientes = mean(yPred) * 100;
numTotalPasajeros = height(testData);
edadMinima = min(XTest(:, 3));
edadMaxima = max(XTest(:, 3));
coeficienteMasBajo = min(abs(B(2:end)));

% Identificar la característica con mayor impacto positivo
[~, idxMax] = max(B(2:end));
features = {'Pclass', 'Sex', 'Age', 'Fare'};

% Mostrar las respuestas a las 10 preguntas en orden
disp('1. Mediana de Age (entrenamiento):');
disp(medianaEdad);

disp('2. Edad promedio (prueba):');
disp(edadPromedio);

disp('3. Característica con mayor impacto:');
disp(features{idxMax});

disp('4. Valor del coeficiente asociado a la variable "Sex":');
disp(B(3));

disp('5. Número total de pasajeros en el conjunto de prueba:');
disp(numTotalPasajeros);

disp('6. Número de pasajeros de primera clase en el conjunto de prueba:');
disp(numPrimeraClase);

disp('7. Porcentaje de predicción de supervivientes en el conjunto de prueba:');
disp(porcentajeSupervivientes);

disp('8. Número de pasajeros predichos como sobrevivientes:');
disp(sum(yPred));

disp('9. Coeficiente más bajo asociado a una característica del modelo:');
disp(coeficienteMasBajo);

disp('10. Edad máxima de los pasajeros en el conjunto de prueba:');
disp(edadMaxima);
