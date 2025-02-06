clear; close all; clc;

%Загрузка данных для классификации аритмии
load arrhythmia
tabulate(categorical(Y));
Table=tabulate(categorical(Y));

rng(2); % начальное значение генератора случайных чисел

dimentions = 20; %20 измерений
%dimentions=dimentions/2;
%dimentions=dimentions/4;
tTree = templateTree('MinLeafSize',dimentions); %Каждый лист имеет как минимум MinLeafSize=20 наблюдения на лист дерева.

Method = 'RUSBoost'; %метод обучения ансабля
NLearn = 100; %- количество циклов обучения ансамбля. На каждом цикле обучаются все деревья шаблона и один представитель слабо обучен для каждого шаблона,
LearnRate = 'LearnRate';

t = templateEnsemble('AdaBoostM1',100, tTree,'LearnRate',0.1) 

Mdl = fitcecoc(X,Y,'Learners',t); % многоклассовая модель
view(Mdl.BinaryLearners{1}.Trained{1},'Mode','graph') %Результаты перекрестной проверки определяют, насколько хорошо модель обобщает
L = resubLoss(Mdl,'LossFun','classiferror')  %дескриптор функции потери или строка, представляющая встроенную функцию потери

