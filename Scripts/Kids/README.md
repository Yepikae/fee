## KIDS ##

### Face classification ###

Issu de ces travaux https://github.com/oarriaga/face_classification qui ont un
taux de réussite de 66% sur la détection d'émotions sur la base FER2013.

#### Pré-traitement ####

* Extraction des points de landmarks via dlib, frame par frame.
* Découpage de chaque frame autour du visage, avec un padding variable.
* Redimensionnement en 68x68.

#### Modèle ####

#### Apprentissage ####

* Utilisation d'un modèle pré-entrainé disponible sur https://github.com/oarriaga/face_classification. Utilisation du modèle 102-0.66
parmis les modèles de détection d'expression.
* Prediction sur l'ensemble des frame issues du prétraitement.

#### Résultats ####

Ci-dessous, les résultats obtenus après prédictions sur les vignettes récupérée
des vidéos. La section **neutral** des colonnes correspond à l'expression
*NEUTRE* telle qu'apprise par le modèle. La section **neutral** des lignes
correspond à une expression indéterminée par les observateur humains.    

| source\target	| anger	    | disgust   | fear	  | happiness  	| sadness  	| surprise  	| neutral   |
| ------------	| --------- | --------- | ------	| ----------	| ---------	| ----------	| --------- |
| anger	        | **64.92**	|     0	    | 0	      | 0	          | 1.93	    | 0	          | 33.15     |
| disgust	      |   16.92	  |   **0**	  | 1.22	  | 16.2	      | 10.38	    | 0	          | 55.28     |
| fear	        |   8.53	  |     0	    | **0**  	| 0	          | 8.29	    | 3.4	        | 79.78     |
| happiness	    |   20.72	  |     0	    | 0.26    | **51.45**	  | 1.11	    | 0.55	      | 25.91     |
| sadness	      |   32.57	  |     0	    | 4.8	    | 2.35	      | **13.82**	| 0	          | 46.46     |
| surprise	    |   30.05	  |     0	    | 0	      | 5.7	        | 8.8	      | **6.88**	  | 48.58     |
| neutral	      |   33.33	  |     0	    | 0	      | 0	          | 0.57	    | 0	          | **66.1**  |

Les expressions NEUTRE et de COLÈRE sont ici surreprésentée. La surreprésentation
de l'expression NEUTRE peu s'expliquer par le fait que chaque *frame*  des
vidéos est transformée en vignette puis classée. Or, les expressions des sujets
observé ne sont pas figées dans le temps, et les vidéos ne sont pas exactement
coupées sur l'expression d'une émotion. D'où la présence d'instants d'expression
neutre.

La surreprésentation de l'expression de COLÈRE peut s'expliquer en partie par
l'angle de la caméra choisi, qui donne une impression de froncer les sourcils
aux sujets d'observation.

Ci-dessous les expressions telles que relevées par 23 observateurs humains. Les
principales différences entre le modèle et les observateurs humains concernent
les expressions de DEGOÛT, de PEUR et de SURPRISE.

| source\target	| anger	    | disgust   | fear	    | happiness  	| sadness  	| surprise  | neutral   |
| ------------	| --------- | --------- | ------	  | ----------	| ---------	| --------- | --------- |
| anger	        | **46.38**	| 8.7	      | 7.25	    | 2.17	      | 10.14     | 4.35	    | 21.01     |
| disgust	      | 1.74		  | **78.26** | 9.57	    | 5.22	      | 0	   	    | 4.35	    | 0.87      |
| fear	        | 6.52	    | 3.99	    | **47.46**	| 0.36	      | 10.87     | 21.74	    | 9.06      |
| happiness	    | 0.8	  	  | 2.4	      | 3.2	      | **77.12**   | 3.2	      | 8.93	    | 4.35      |
| sadness	      | 6.28		  | 2.9	      | 11.6	    | 5.31	      | **52.17**	| 5.8	      | 15.94     |
| surprise	    | 5.07		  | 5.22	    | 12.9	    | 4.78	      | 1.45	    | **64.06** | 6.52      |
| neutral	      | 11.6		  | 2.9	      | 24.64	    | 1.45	      | 13.04     | 1.45	    | **44.93** |


### LSTM ###

Nous vons combiné le modèle précédent à un Long Short Term Memory afin de
tenir compte des animations pour classer els expressions. L'idée est de créer des
séquences de *n* images issue d'une vidéo, d'en extraire les *features* via le
modèle vu précédement puis d'apprendre sur les séquences de features.

#### Pré-traitement ####
* Extraction des points de landmarks via dlib, frame apr frame.
* Découpage de chaque frame autour du visage, avec un padding variable.
* Redimensionnement en 68x68.
* Grouper par *n* images successives.

#### Modèle ####

#### Apprentissage ####

* Première passe d'apprentissage sur les vidéo de CK. (les vidéos sont
  pré-traitées de la même manière que celles de kids).
* Prédiction sur les vidéos de kids.

#### Résultats ####

Ci-dessous, les résultats obtenus après prédictions sur les séquence récupérée
des vidéos. La section **neutral** des colonnes correspond à l'expression
*NEUTRE* telle qu'apprise par le modèle. La section **neutral** des lignes
correspond à une expression indéterminée par les observateur humains.    

| source\target	| anger	    | disgust   | fear	   | happiness | sadness | surprise  | neutral   |
| ------------	| --------- | --------- | ------	 | --------- | ------- | --------- | --------  |
| anger	        | **22.35**	|     0	    | 0	       | 0	       | 0.93	   | 0	       | 76.72     |
| disgust	      |   0.56	  |   **3.2** | 10.13    | 20.57	   | 0	     | 5	       | 60.54     |
| fear	        |   16.95   |     0	  	| **9.88** | 2.45	     | 9.26	   | 2.5	     | 58.96     |
| happiness	    |   4.43	  |     0.16	| 1.89	   | **56.14** | 0.06	   | 3.96      | 33.35     |
| sadness	      |   26.05	  |     0	  	| 0.6	     | 0	       | **0.6** | 7.35      | 65.4      |
| surprise	    |   16.1	  |     0	  	| 3.16	   | 10.9	     | 0.39	   | **12.73** | 56.72     |
| neutral	      |   40	 	  |     1.75	| 4.01	   | 2.56	     | 0	     | 0	       | **51.67** |

La surreprésentation de faux-positifs sur la classe de COLÈRE a largement diminué.
L'expression NEUTRE reste quant à elle surreprésentée.

Dans un second temps, nous avons ignoré les expressions prédite NEUTRE et pris
la seconde expression le cas échéant. LE tableau ci-dessous représente les
résultats obtenus.

| source\target	| anger	    | disgust   | fear	    | happiness | sadness | surprise  |
| ------------	| --------- | --------- | ------	  | --------- | ------- | --------- |
| anger	        | **71.3**	|     0	    | 6.83	    | 10.1	    | 1.42	  | 10.35     |
| disgust	      |   5.43	  |   **3.2** | 22.59	    | 46.26	    | 3.89	  | 18.63     |
| fear	        |   36.82   |     0	  	| **20.66**	| 15.2	    | 13.35	  | 13.97     |
| happiness	    |   8.18	  |     0.16	| 7.21	    | **74.32** | 1.58	  | 8.55      |
| sadness	      |   50.49	  |     0	  	| 14.19	    | 6.73	    | **4.5**	| 24.08     |
| surprise	    |   30.64	  |     0	  	| 21.52	    | 14.52	    | 2.57	  | **30.75** |
