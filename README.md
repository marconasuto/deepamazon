
# deepAmazon
*How deep learning and satellite images can help tracking and planning a sustainable strategy for a region.*

*Topic: CONVOLUTIONAL NEURAL NETWORKS APPLIED TO REMOTE SENSING IMAGERY FOR MULTI-LABEL SCENE RECOGNITION.*

*Module 5 project for FlatIron School Data Science course*

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "AMAZON BASIN MAP: FIRES, MINING, OIL AND INDIGENOUS AREAS (UPDATED 2019)")
FIG AMAZON BASIN MAP: FIRES, MINING, OIL AND INDIGENOUS AREAS (UPDATED 2019) 
DESCRIPTION: IT CAN BE EASILY SEEN THAT THERE IS A GOOD OVERLAP BETWEEN FIRES AND MINES SITES LOCATIONS. 
SOURCE: NASA/NOAA, VIIRS DAILY GLOBAL FIRE DETECTIONS; DETECTIONS: AMAZONIAN NETWORK OF GEOREFERENCED SOCIOENVIRONMENTAL INFORMATION (RAISG); CREDITS: MARCO GIUSEPPE NASUTO

## Project structure
```
├── LICENSE
├── README.md                                                <- The project layout (this file)
├── data
│   ├── images                                                   <- For README.md, presentation
│   ├── train                                                        <- Training set
│   ├── val                                                          <- Validation set
│   ├── test                                                         <- Test set
│   └── raw                                                         <- The original, immutable data dump
│                  └── geospatial                               <- Geospatial data
│                                └── petroleo2020             <- Oil sites
│                                └── Mineriallegal2020      <- Illegal mining sites
│                                └── Tis_2020                    <- Indigenous population territories
│                                └── mineria2020               <- Mining sites
│                   └── chips                                       <- Satellite chips 
│                                └──jpg                               <- jpg format 
│                                └── tif                                <- tif format (4 bands)
│                    └── chips.csv                                <- Chips labels table reference
│
├── load_eda_train.ipynb       <- Input, training, validation, test, reports
├── utils.py                            <- import libraries and utility functions
│
├── reports                              <- Reports and presentations
│   └── presentation.pdf         <- Non-technical presentation
│   └── assets                         <- Plots
│   └── logs                             <- Tensorboard logs
│   └── saved_models             <- Saved models
│   └── maps                           <- Folium map in HTML
└── requirements.txt                <- The requirements file for reproducing the analysis environment
```

## Project description
The strategic development planning of a region is a [highly interdisciplinary task](https://jssidoi.org/ird/uploads/issues/Insights_into_Regional_Development_Vol1_No3_print.pdf) that requires a set of tools,  competencies, people and timing, to come up with a political vision of the future of a region. Any modern development strategy should be thought within the [sustainable development framework](https://www.un.org/sustainabledevelopment/blog/2016/03/un-statistical-commission-endorses-global-indicator-framework/), which has at the core the concept of [sustainability](http://www2.econ.iastate.edu/classes/tsc220/hallam/TypesOfSustainability.pdf) (human, social , economical, environmental). The field is obviously very complex due to internal, regional conditions, and external factors, as regions are dynamic, interconnected systems. A thorough analysis require both an enormous, etherogeneous (as the problem is multi-faceted) amount of data, and a substantial degree of synthesis of this information, in order to distill insights and make them actionable. Machine learning offers the chance and the hope to leverage on this data and transform it into information. Often digitalization is misconcieved as de-materialization, something closer to magic than reality. Every digital activity has a physical aspect/process somewhere along its supply chain, that can be described by a set of features and, by being physical, it can be  displayed on a map. Remote sensing data is key for geospatial analysis, especially within the context of strategic development planning of a region. One of the building blocks of a system able to track and analyse objects from RS images, is to detect and classify them. Although in rapid development, deep learning, a subfield of machine learning that showed impressive applications on computer vision tasks, has not being applied extensively to remote sensing images with respect to other type of imageries, yet.

One of the most symbolic, endangered areas, where strategic sustainable development planning is not only key, but has also an impact on a global scale, is the Amazon basin. Since the greatest majority of the basin belongs to Brazil, I will focus on the Brazilian side. Brazilians have always seen the Amazon basin as a potential, enormous resource of wealth. Due to strategic reasons, the extension, the natural resources, and at the same time the impenetrability of the rainforest (seen as an hazard from a military point of view), were subject to mega-projects. This type of development model has not only devastating downsides on the environment, but also shows economical criticalities in terms of growth. Alternatives to mega-projects have been implemented only in the last 25 years. Sustainable development follows different paradigms, the main one is a slower but more stable growth rate. An interesting and successful example of sustainable business in the Brazialian Amazon, is the project [Castanha-do-Brasil](https://www.scielo.br/scielo.php?script=sci_arttext&pid=S0101-31572004000200306&lng=en&nrm=iso#fn17). In the Business understanding section, I did a short literature review on the topic. 

In 2017, [Planet](http://planet.com/), designer and builder of the world’s largest constellation of Earth-imaging satellites, launched a challenge on [Kaggle](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space), offering an interesting dataset of satellite images of the Amazon basin with a 3 - 5 meter resolution. There are 17 unbalanced labels in total: 4 labels for the atmospheric conditions, 13 labels land cover/land use. Technically, the task of this project consists of building a multi-label, land cover/land use and atmospheric conditions classifier (in other words a scene classifier) leveraging on deep learning architectures, in particular CNNs.

In September , 2020 became the [most active](http://globalfiredata.org/pages/) fire year in the southern Amazon since 2012*(the year that the VIIRS sensor was launched on the Suomi-NPP satellite). The majority of fires occur due to regional deforestation. In August 2019, a research by the National Institute for Space Research (INPE) revealed an [84 percent increase](https://www.bbc.com/news/world-latin-america-49415973) in Amazon forest fires compared to 2018. Bolsonaro and its establishment [fired the Head of INPE](https://www.nytimes.com/2019/08/02/world/americas/bolsonaro-amazon-deforestation-galvao.html) due to tracking Amazon deforastation.

After framing Business understanding, I scoped the project to offer recommendations to the the following initial questions: **1) Can help tracking the exploitation of the Amazon basin in a more efficient way (i.e. less manual working)?** **2) Where are located the majority of fires?** **3)** **Can this tool be potentially used for verifying deforestation alerts?** . After doing a literature review, that can be found in [Related works](https://marconasuto.com/deepamazon), I did a quick Exploratory Data Analysis on the dataset first. To enrich the analysis (map above and fig. 1) , I also used RS [VIIRS I-Band 375 m Active Fire Data](https://earthdata.nasa.gov/earth-observation-data/near-real-time/firms/viirs-i-band-active-fire-data) on active fires in 2019 in Brazil, and several geospatial data on the Amazon basin (updated to 2020) from [Amazonia Socioambiental](https://www.amazoniasocioambiental.org/en/maps/#download),  to create a map of the Amzon basin with different data layers to get a quick overview.

Once done with the EDA, 1) I created a baseline CNN, adding levels of complexity through some iterations (i.e. input shape, number of samples, depth and complexity of the model, optimizers etc.), based on training and validation metrics, in my case f2-score and loss. 2) I used a pre-trained architecture ([ResNet50 as it offer a good trade-off](https://arxiv.org/pdf/1810.00736.pdf) between accuracy and computational complexity), unfreezing first only the top layers (fine-tuning), than trying training from scratch on the entire training set .

Due to strong computational and time constraints (as part of the project requirements, the final scripts must be ran at max overnight on a local machine - in my case a MacBook Pro mid 2015), results and accuracy were far from Kaggle level results. Neverthless, I tried to focus and start learning using Tensorflow (2.4.1) tf.data pipeline and Tensorflow low level instead of just using Keras.

In the initial Kaggle competition, ranking was based on average F2 score, which is a Fbeta score 'focued' on recall, hence on minimisizng the number of false negatives. I approached the problem doing a literature review on several topics, rather than going 'plug-and-play' approach.

Finally, I summarised the results, limits of this project and highlighted some of the possible future works.

## Business understanding
### SDGs and business opportunities: a brief overview
According to the [United Nations](https://www.earthobservations.org/documents/publications/201703_geo_eo_for_2030_agenda.pdf), Earth Observation and Remote Sensing data are crucial for reaching the Sustainable Development Goals (SDGs). In 1972, [The Limits to Growth](https://www.clubofrome.org/publication/the-limits-to-growth/) was published by the Club of Rome. Levereging on the MIT computer World3, it was a study on the relationships between five subsystesms of the world economy population, food and industrial production, pollution and consumption of non-renewable natural resources. The key finding was that an unlimited growth in the economy and population would lead to a collapse of the global system by the mid to late twenty-first century. In 2012, the United Nations published the Sustainable Development Goals that set what the most significant challenges for the next two decades are. Earth Observation data can help tackling all of them (fig.2), at a significant scale, with a great level of granularity, and at low cost. 

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "MAPPING OF HOW EARTH OBSERVATION DATA CAN HELP ACHIEVING SUSTAINABLE DEVELOPMENT GOALS FOR 2030 AGENDA. SOURCE: GEO - UNITED NATIONS")
FIG 3: MAPPING OF HOW EARTH OBSERVATION DATA CAN HELP ACHIEVING SUSTAINABLE DEVELOPMENT GOALS FOR 2030 AGENDA. SOURCE: GEO - UNITED NATIONS

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "CLASSIFICATION TASKS ACHIEVABLE USING MACHINE LEARNING APPLIED TO RS DATA TOWARDS SUSTAINABLE DEVELOPMENT GOALS. CREDITS: FERREIRA, ITEN & AL. ")
FIG 3: CLASSIFICATION TASKS ACHIEVABLE USING MACHINE LEARNING APPLIED TO RS DATA TOWARDS SUSTAINABLE DEVELOPMENT GOALS. CREDITS: FERREIRA, ITEN & AL. 

**Food and agriculture**
It is estimated that roughly [800 million](https://www.wri.org/insights/numbers-how-business-benefits-sustainable-development-goals) people are undernourished globally. The growing population will require 220 million hectares (approximately the size of Ireland) additional land for food production. On the other hand it has been estimated that from 1990 till today, [220 million hectares of tropical forests disappeared mainly due to agriculture](https://advances.sciencemag.org/content/7/10/eabe1603), logging and fires. Considering that deforestation is linked with climate change due to CO2 absorption, and that [according to McKinsey](https://www.mckinsey.com/industries/oil-and-gas/our-insights/global-energy-perspective-2021), we're far from the 1.5º C pathway, it is evident that food and agriculture pose a serious challenge to environmental sustainability. It might be a clichè, but one side of big challenges is that they offer big opportunities:

The top three opportunities are food waste reduction, reforestation and development of low-income food markets. The estimated porential opportunity in revenues is in the order of magnitude of trillions of  dollars (USD) in revenues.

**Cities**
At the moment half the global population resides in cities. By 2050 the figure will reach 70 percent of the world population. The two main challenges of cities in terms of sustainability are: energy use and poverty. SDGs and their achievements offer an economic opportunity of trillions of dollars, mainly in the sectors of building and transportation.

**Energy and Materials**
Circular models can positevly impact energy and material consumptions in terms of trillions of economic opportunities. The two main areas of development are energy supply chain and recovery of materials from waste.

**Health and Well-Being**
An unsustainable world affects th healthcar syste The benefit of achieving SDGs can be translated not only in terms of well-being, but  lso as business opportunity. The adoption of advanced technolgies goes beyond a better service for the citizens, it also means scalable solutions, access and optimization of resources. Sustainable healthcare is another trillion-figure business opportunity.

### Sustainable development of the Amazon basin: practical cases
As environmental awareness has grown in the last decades, [pressures](https://www.nytimes.com/2019/09/05/world/americas/h-m-leather-brazil-amazon-fires.html) over the different actors operating directly or indirectly on the Amazon basin (private sectors and governments) manifested in several attempts and calls for action on changing a mainly predatory supply chain system, which is unsustainable from an environmental point of view, as well as from an economic point of view in the long term. This predatory mechanism is called boom and bust economic cycle. [The lack of control](https://science.sciencemag.org/content/361/6407/1108) of the supply chain, makes even corporate initiatives difficult to be executed. 27% of the global deforastation happens due to permanent land use change due to commodity production. Machine learning is already used to help tracking deforestation. [GLAD](https://glad.umd.edu/projects/global-forest-watch), Global Land Analysis Discovery, uses satellite images (Landsat) and decision tree classifier algorithm for its project Global Forest Watch. Deep learning, however, hasnt been adopted, yet.

There are two main models of development of the Amazon: one is represented by mega-projects, the other one by smaller scale, sustainable projects. Brazilians have always seen the Amazon as an enormous resource of potential wealth. Generally speaking, it's one of the poorest areas of the country. Historically, the presence and the size of the Amazon rainforest has been seen as problem and resource of strategic importance by the Brazilian state. Until 1990s, the only model of development was mega-projects, which consist of the realization of massive infrastructures (dams, roads, power plants) that should accelerate the industrialization of the Amazon, hence its economic growth. They usually require significant amount of capital injection, public debt, foreign investments, and they involve a lot of workers. The decision-making from ideation to fullfilment is usually centralized, without involving the local best practices.

The main advantages are in terms of the scale of the economic growth and jobs. Both these aspects are however overesitmated in the long term. After completion, the workers are not needed in the area where the project was realised, the environmental costs are tremendous and even useless. Indeed, studies have shown that due to average annual rainfall levels, only 17% of the Amazon basin has a chance of economic success through agriculture and ranching. The greatest majority of the Amazon territory requires an approach that has at its core a susitanable development of the forest. Why "sustainable" and not just a more generic "development"?

Predatory economic practices in the Amazon, like selective logging, have been demonstrated as generating rapid economic growth in the short term, high environmental costs, followed by sharp falls in growth and employement, leading to an unstable economy and tax base.

Small scale, sustainable models have their strength in a,although lower in terms of magnitude, stable, inclusive (i.e. gender equality)  economic growth,with significant benefits in building citizenship, while preserving the environment. One of the best and most successful examples of this type of developm,ent model is the [Projeto Castanha-do-Brasil](https://www.scielo.br/scielo.php?script=sci_arttext&pid=S0101-31572004000200306&lng=en&nrm=iso#fn17). Key aspects of this case (started only in 1995) are:

The focus on a natural, local resource (Brazilian nuts), tied to its sustainable exploitation both from an environmental and a social point of view (castanheros were at the lowest level of the social ladder); 
Women are structural actors the economic success of the project
The government role as an initiator, main commercial partner (although now more than 50% of the revenue stream comes from non-governamental customers), and regulator;
The legal and social form of the companies part of the project was co-ops, not big corporates, with a re-distribution (partial)/welfare factor embedded in the economic fabric;
Verticalization and control of the supply chain, from harvesting to commercialization and spin-offs activities (all the products derivated from the manufactoring of the Brazilian nut, from food products, i.e. cookies, to pharmaceutical products); 
Scientific research and development involving  Institute for Studies and Investigation (IEPA)
A political vision that rather than focusing on maximising growth in the short term, devastating the social and environmental fabric, bets on a slower, steadier and more inclusive growth.

The limits and criticism of cases like Castanha-do-Brasil project, are mainly the scale of the growth, the pace and thedependency from the government, especially during the first decades/phase. With no real economic growth, the local-driven economic programs are nothing more than social programs. However, if we take the case of sustainable logging, the initial growth is slower than the one operating with a approach,  but jobs and profits remain constant long after the drop-off would have occurred creating a stable economy, a stable work environment, and a stable tax base.

Considering the described scenarios, context and business opportunities from sustainable development, our project :

- Can help tracking the exploitation of the Amazon basin in a more efficient way (i.e. less manual working)?
- Where are located the majority of fires?  
- Can this tool be potentially used for verifying deforestation alerts?

### Project level maturity w.r.t. Earth Observation service provider maturity
The service offered by this project, contextualized w.r.t. the market of  Earth Observation service providers, could potentially return has a high level of benefits to users (i.e. governamental entities). Moreover, if we frame this technology as a necessary building block for the development more advanced tools with quantification and prediction features, its added value increases. 

Last but not least, although not an easy, immediate next step in a potential features development roadmap,  this tool could be generalized to other geographic areas, esponentially increasing its flexibility and application power.

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "CO-OCCURENCY MATRIX FOR ALL LABELS")
FIG: PROJECT MATURITY W.R.T. EARTH OBSERVATION SERVICE MATURITY. GRAPH BASED ON [PLANETEK ITALIA](https://www.youtube.com/watch?v=uprTuje90vk&t=1018s) SPACE BUSINESS MODEL SLIDES.

## Related work 
### Classification scene methods in Remote Sensing 
The first paridgm in RS imagery classification was at single pixel level. [Blaschke and Strobl](https://www.researchgate.net/publication/216266284_What's_wrong_with_pixels_Some_recent_developments_interfacing_remote_sensing_and_GIS) work summarised a shift within RS and GIS community, where, rather than pursuing the statistical analysis of single pixels, they stressed the necessity to focus on the spatial meaningful semantic entities or scene components, distinguishable in an image through patterns built up by pixels: in other words "objects".

The object-level paradigm have dominated RS analysis for decades. However, limitations of methods based on this paradigm are in relatively poor semantic meanings performance, with respect to more recent techniques.

Semantic-level has been the new paradigm for high resolution RS scene classification which aims to label each scene, which are parts of images,  with a specific semantic class (i.e. agriculture, mining, road, water etc.).

Classification methods can be divided in three main categories, not necessarily independent,  depending on the features used: 1) handcrafted feature based methods; 2) unsupervised feature learning based methods; 3) deep feature learning based methods. I will focus on the latter.

Most of the current state-of-the-art approaches generally rely on supervised, deep learning methods, due to their good capabilities at discovering non-linear and complex relationships hidden in high-dimensional data. In the last years, there have been major improvements and experiments on different deep learning architectures applied to scene classification, among them deep belief nets (DBN), deep Boltzmann machines (DBM) , stacked autoencoder (SAE) , and convolutional neural networks (CNNs).

Multi-spectral images are an array of X rows, Y columns and P spectral channels (bands). CNNs are designed to process data in the form of multiple arrays. Just to mention a few of the recent, most famous, CNNs architectures: [AlexNet](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf), [VGGNet](https://arxiv.org/abs/1409.1556), [GoogLeNet](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43022.pdf), [ResNet](https://arxiv.org/abs/1512.03385), [DenseNet](https://arxiv.org/pdf/1608.06993.pdf), [YOLO](https://arxiv.org/pdf/1506.02640.pdf), [RetinaNet](https://arxiv.org/pdf/1708.02002.pdf).

A CNN consists of series of layers. Feature extraction occurs in convolutional layers, which consist of convolving inputs with a set of filters followed by an activation function to add non-linearity. The first convolutional layers capture lower level features (i.e. edges) whereas the deeper layers extract more abstract and complex features by combining low-level layers. In order to reduce the number of training parameters, computational cost and overfitting, after a certain number of convolutional layers, pooling layers are another key component of CNNs. Overfitting, hence generalization, can be achieved using regularization layers (i.e. L1 and L2 regularization) and dropout layers, by randomly subsampling the outputs of a layer. Another key component of CNNs are normalization layers. By normalizing inputs after a convolutional layer or a fully connected layer [before the activation function](https://arxiv.org/pdf/1502.03167.pdf),  normalization layers prevent instability and vanishing/exploding gradients, although, recently it  has been proved that actually using batch normalization [after the activation](https://arxiv.org/abs/1905.05928), togheter with dropout layers helps achieving better performance. Finally, the 'head'of a CNN is usually some fully connected (FC)  layers with a final FC as a classifier.

It is not only possible but actually advisable, especially for small datasets and strong computational limitations, adopting deep transfer learning techniques,  consisting in using existing CNNs, for scene classification of remote sensing images. The success of this approach depends on several factors, the most important one being the similarity between the original task on which the CNN was originally trained and the target task. I.e. using CNNs trained on the ImageNet dataset for multi-spectral RS imagery makes sense due to low-level similarities with general-purpose images. On the other hand, the same [would not apply to SAR](https://arxiv.org/pdf/1508.00092.pdf) (radar) images due to their peculiar pixel-level statistics. The three main trasnfer learning strategies are: full training, fine tuning, and feature extraction. [Feature Extraction](https://my.readymag.com/edit/2753908/preview/) consists in using features learned by a previous network, removed the classifier layer, to detect  features from new samples, by adding a new classifier, that will be trained from scratch, on top of the pretrained model (i.e. a linear classifier, SVM etc.). Therefore, retraining the entire model is not needed. Fine-Tuning consists in de-freezing some of the layers of a pre-trained network, adding a new classifier on top and train/fine-tune the weights of the trainable layers. This allows us to "fine-tune" the more abstract  feature representations on the specific target task. [Experimental results](https://www.mdpi.com/2072-4292/12/1/86) show that fine tuning tends to be the [best performing strategy](https://arxiv.org/pdf/1703.00121.pdf) on small-scale datasets.

The loss functions considered were binary crossentropy, which is common for multi-label classification. Ideally, since the target metric is F2, the loss should directly optimize it. Unfortunately, F-beta score is not differentiable, making it unusable for stochastic gradient descent (which instead is an optimizaion technique for convex functions). As implemented both by [Google](http://proceedings.mlr.press/v54/eban17a/eban17a.pdf) and Microsoft researchers, non differentiable losses can be 'transformed' into differentiable ones via surrogated losses. The implementation of the latter was also used by the winner of the competition on this dataset on Kaggle. I also wanted to experiment with the focal loss, proposed by Facebook AI Research (FAIR) in their work on Dense Object Detection. Unfortunately, there wasnt enopugh time to play around with it, therefore I added it as something for future works.

## Dataset
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "COUNT PLOT OF NUMBER OF IMAGES PER LABEL")
FIG : COUNT PLOT OF NUMBER OF IMAGES PER LABEL

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "CO-OCCURENCY MATRIX FOR ALL LABELS")
FIG : CO-OCCURENCY MATRIX FOR ALL LABELS

The Kaggle [dataset](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space) that I used in this project is a set of 40,479 satellite imagery of the Amazon Basin that was collected over a 1-year span starting in 2016. The dataset was provided by Planet. Each image is 256x256x4 in a 4-channel(RGB+near-IR) TIFF format. I had also to download the dataset with images in jpeg format, as explained later. There are 17 classes andeach image can belong to multiple classes. The labels canbroadly be broken into three groups: weather conditions,common land cover/land use, and rareland cover/land use. The weather conditionslabels are: clear, cloudy, partly cloudy, and haze. The commonland labels are: primary, agriculture, cultivation, habitation,water and roads. The rare labels are: slash-and-burn, selectivelogging, blooming, bare ground, conventional mining,artisinal mining, and blow-down. It is evident from the distribution of labels across training images (fig.2) that this project presents challenges due to class imbalance. Fig.3, 4, 5, show co-occurence matrices of labels, which help understanding some patterns: weather labels are mutually exclusive; commonlabels have heavy overlap whereas rare labels have very minimal overlap. In addition, we calculated the normalized difference vegetation index (NDVI) and green NDVI (GNDVI) for all datasets as the inclusion of vegetation indices [previously](https://arxiv.org/pdf/1709.06764.pdf) has been shown to improve the performance of CNN models applied to UAV orthogonal images, trained on small datasets. NDVI and GNDVI were derived from image bands using the following equations:
IMAGE

Preprocessing encompassed data augmentation: 90 deg rotation, left-right flipping; one-hot-encoding of labels; image parsing included decoding, scaling either using ResNet preprocess unit  or dividing each pixel by 255; resizing.

The [data](https://earthdata.nasa.gov/earth-observation-data/near-real-time/firms/viirs-i-band-active-fire-data) about fires refer to the entire 2019 and were spotted by VIIRS 375m sensor, on board of the Suomi NPP/NOAA-20 satellite. VIIRS has a thermal band of 375m resolution, w.r.t., i.e. MODIS thermal resolution of 1000m.VIIRS spotted more than 1.45M fires only in Brazil. The dataset consists of all fires detected, [reporting](https://earthdata.nasa.gov/earth-observation-data/near-real-time/firms/v1-vnp14imgt) their geolocations, information about the acquisition (scan, track, refers to the actual size of the pixel, date, daynight about the time), the instrument and satellite used, they measured brightness in the I-4 and I-5 channels, and the type of fire I filtered the fires by type and confidence level (only vegetation fires and with confidence Nominal' or 'High'. Plotting requried downsampling the number of points/fires as they are in the order of magnitude of millions. I randomly downsampled, between the IQR range, the 5% of the total number and plotted on the map.

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "VIIRS DATASET FIRES SPOTTED IN BRAZIL DURING 2019")
FIG : VIIRS DATASET FIRES SPOTTED IN BRAZIL DURING 2019

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png " BAR PLOT OF TYPE OF FIRES SPOTTED BY VIIRS 375M IN BRAZILE IN 2019")
FIG : BAR PLOT OF TYPE OF FIRES SPOTTED BY VIIRS 375M IN BRAZILE IN 2019; NOTE THAT THE ALMOST TOTALITY IS VEGETATION FIRE


![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "HISTOGRAM OF VEGETATION FIRES SPOTTED BY VIIRS 375M IN BRAZIL BY THEIR RADIATIVE POWER(MW) - 2019")
FIG : HISTOGRAM OF VEGETATION FIRES SPOTTED BY VIIRS 375M IN BRAZIL BY THEIR RADIATIVE POWER(MW) - 2019 (SOURCE: SUOMI NPP SATELLITE/NASA); THESE ARE THE FIRES LABELLED WITH CONFIDENCE NOMINAL OR HIGH.

I added data about some of the human activities with the highest environmental impact, such as oil and mining sites, as well as indigeneous population areas. The data was downloaded from the Amazonian network of georeferenced socioenvironmental inforamtion (RAISG). I I filtered by country (Brazil) and randomly downsampled due to the order of magnitude of points that would have been plotted on the map.

## Methods

The methodology behind this project can be summarised in: Business understanding, Related works research and the standard OSEMiN (Obtain, Scrub, Explore, Model, Interpret). I started with Keras API ImageDataGenerator, moving to tf.data, and finally experimenting with lower level Tensorflow (2.4.1), implementing training and evaluation steps. Eager execution was active, as this project is not meant to go into production. No GPUs or TPUs were used for training or prediction.

I faced several issues with the different decoding, color spaces and rescaling of tiff and jpeg images.

 Reading tiff: tiff images are 16-bit multi-spectral images, the band order is blue, green, red, near IR (bgra). Depending on the library adopted to read them, things can get tricky with the channels order. Let's start from reading a tiff image using skimage, aka scikit-image, and visualizing using SPy (spectral python), package for hyperspectral images:

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "COUNT PLOT OF NUMBER OF IMAGES PER LABEL - TRAINING SET")
FIG:

that returns:

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "COUNT PLOT OF NUMBER OF IMAGES PER LABEL - TRAINING SET")
FIG:

checking the array values:

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "COUNT PLOT OF NUMBER OF IMAGES PER LABEL - TRAINING SET")
FIG:

OpenCv reads images as bgr(a). However, being tiff images channels order bgra, OpenCv, de facto, changes the order to rgb(a). 

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "COUNT PLOT OF NUMBER OF IMAGES PER LABEL - TRAINING SET")
FIG:

PIL, which is used in tensorflow ImageDataGenerator, reads images as rgb. It returns an array decoded in 8-bit (uint8). When reading tiff, it actually outputs a bgr image:

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "COUNT PLOT OF NUMBER OF IMAGES PER LABEL - TRAINING SET")
FIG:

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "COUNT PLOT OF NUMBER OF IMAGES PER LABEL - TRAINING SET")
FIG:

Starting from how PIL (hence ImageDataGenerator) reads tiff images, I'll move to the second issue I faced.

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "COUNT PLOT OF NUMBER OF IMAGES PER LABEL - TRAINING SET")
FIG

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "COUNT PLOT OF NUMBER OF IMAGES PER LABEL - TRAINING SET")
FIG

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "COUNT PLOT OF NUMBER OF IMAGES PER LABEL - TRAINING SET")
FIG


2. Rescaling intensity values: data pipelines in tensorflow can be implemented in different ways. I wanted to use tf.data w.r.t. ImageDataGenerator, as tf.data offers faster performance. Tensorflow offers an experimental function called tfio.experimental.image.decode_tiff to decode string tensor (returned by tf.io.read_file) to a uint8 array, with intensity values rescaled to 0-255. I'll show you the differences in terms of rescaling intensities amid PIL images, tensorflow decode_tiff and opencv, the latter opening only the first 3 channels:

As we can see, the intensity values in the PIL image are significantly different from the others, something noticeable when visualizing the images. This rescaling issue becomes of critical importance when training the model, with a big drop (10-15% order of magnitude) in terms of performances and learning (loss optimization). 

Unfortunately, I couldn't manage to fit skimage.io module with tf.data pipeline. One of the solutions tried consited in converting tf.io.read_file tensor bytes into an image. But when decoding bytes, the shape is not 256x256x4, so I cannot reshape it into an image, and I couldn't find what the extra elements in the array correspond to. A possible alternative was to get the path of each filename using .numpy() tensorflow method. Running in eager_execution enabled, this doesn't work when creating the tf.data dataset.

I had to (painfully, considering the amount of time trying to debug this tensorflow limitation) switch to the jpeg version of the Planet Kaggle dataset. However, I found out, when trying to parse via tf.io.decode_jpeg, that the tensorflow detects incorrectly the color space of those images, returning arrays of zeros.

I had to preprocess the entire dataset using ImageMagick to correctly edit the color space of the images so that tensorflow can read it. The result is achievable running the following line in the shell, once in the images folder:

mogrify convert -colorspace sRGB "*.jpg"

3. The importance of the right input channels order when using pre-trained models: As stated in Keras ResNet official guide, rescaling should be done via the resnet_preprocess. What it does is converting the images from RGB to BGR, then zero-centers each color channel with respect to the ImageNet dataset, without scaling (this is called Caffe style). This means that, depending on the file format and the parser (opencv, PIL, skimage, tensorflow) we need to be careful at the channels order before feeding  resnet_preprocess.


Moving on my experiments with training, I first tried with training from scratch variations of a cascade architecture, with an increasing number of layers and complexity. Then I experimented several iterations of a ResNet50 as a feature extractor, using ImageNet weights, with a GlobalAveragePooling layer, a dropout layer and  a final classifier consisting of 17 neurons with sigmoid activation. This architecture was taken from Keras ResNet official page. I chose ResNet50 (the smallest of the ResNet family) as it is a good trade off between computational complexity and accuracy.  Finally, I fine-tuned the ResNet50 model, playing with increasing the number of  layers trainable, until fully training all  the layers. 

Methodologically, I took decisions on how to improve performance based on two graphs: loss curves and f2 score curves for training and validation sets. 

Due to computational and time constraints, I downsampled the original dataset to 30%, with a number of epochs between 8 and 10 for all quick iterations (1h each). Finally, once I found the best configuration, I trained on the full dataset. Train-test split ratio and train-val ratio were respectively 0.9 and 0.8. Train, validation and test sets were obtained with random shuffling and sampling. Distributions show that validation and test sets are representative of the initial sample distribution in terms of labels. I didn't use cross fold validation due to time constraints.

The hyperparameters tuned were: loss function: binary cross-entropy, soft f2 score; optimizer: SGD,  Adam; number of epochs: 8, 10, 20; number of layers; learning rate: 10ˆ-4 with ReduceLROnPlateau callback to 10ˆ-6; dropout rate: 0.2. 

The evaluation metric adopted was f2-score.

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "COUNT PLOT OF NUMBER OF IMAGES PER LABEL - TRAINING SET")
FIG : COUNT PLOT OF NUMBER OF IMAGES PER LABEL - TRAINING SET OBTAINED WITH TRAIN-TEST SPLIT 0.9, AND TRAIN-VAL SPLIT 0.8, RANDOM SHUFFLING AND SAMPLING

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "COUNT PLOT OF NUMBER OF IMAGES PER LABEL - VALIDATION SET ")
FIG : COUNT PLOT OF NUMBER OF IMAGES PER LABEL - VALIDATION SET OBTAINED WITH TRAIN-TEST SPLIT 0.9 AND TRAIN-VAL SPLIT 0.8, RANDOM SHUFFLING AND SAMPLING

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "COUNT PLOT OF NUMBER OF IMAGES PER LABEL - TEST SET")
FIG : COUNT PLOT OF NUMBER OF IMAGES PER LABEL - TEST SET OBTAINED WITH TRAIN-TEST SPLIT 0.9, RANDOM SHUFFLING AND SAMPLING


## Results

I used Keras with Tensorflow backend for all our experiments. The 4-channel TIFF images perform much worse than the 3-channel JPG images. As discussed in several Kaggle discussions, this is due to calibration issues and mislabeled images. For the transfer learning models, we see that using more powerful models improves results.

Computational and time constraints significantly limited the scope and the level of accuracy of this project. I iterated on the cascade baseline models, adding complexity (conv and fc layers) and using dropout layers as regularization, based on the anlaysis of the loss and f2-score curves.

I'll start presenting the results from the baseline models, more for the sake of the analysis, in other words to explain the curves in the figures, than for the actual perforamnce, which is expectedly poor. As from literature review, with small datasets (such as this one), the most powerful approach is to use transfer learning, in particular fine tuining. Nevertheless, the figures of baseline models refer to the training and validation set downsampled to 7500 images and 3200 images respectively. The baseline models are described in the following code:


```def create_model(version, params):

if version == 'v1.0':

    # Baseline
    
    inputs = Input(shape=(params.size, params.size, params.channels))

    x = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(params.num_classes, activation='sigmoid')(x)  
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss=train_params.loss_obj, optimizer=params.optimizer_obj, metrics=[f2_score])

    

    

if version == 'v1.1':

    # v1.0 with 128 units in FC layer w.r.t 64 

    inputs = Input(shape=(params.size, params.size, params.channels))

    x = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(params.num_classes, activation='sigmoid')(x) 
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=train_params.loss_obj, optimizer=params.optimizer_obj, metrics=[f2_score])



if version == 'v1.2':

    # v1.1 with dropout layers after each block
    
    inputs = Input(shape=(params.size, params.size, params.channels))

    x = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)

    outputs = Dense(params.num_classes, activation='sigmoid')(x)  
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss=train_params.loss_obj, optimizer=params.optimizer_obj, metrics=[f2_score])

if version == 'v1.3':

    inputs = Input(shape=(params.size, params.size, params.channels))

    x = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)

    outputs = Dense(params.num_classes, activation='sigmoid')(x)  
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(loss=train_params.loss_obj, optimizer=params.optimizer_obj, metrics=[f2_score])

    

if version == 'v1.4':

    inputs = Input(shape=(params.size, params.size, params.channels))

    x = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)

    outputs = Dense(params.num_classes, activation='sigmoid')(x)  
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss=train_params.loss_obj, optimizer=params.optimizer_obj, metrics=[f2_score])

return model
```

First: given the size of the datasets (small) and the task, multi-label classification with 17 highly imbalanced classes, what we can expect from models trained from scratch is that their micro-f2 score performance will probably be poor. Micro metrics are computed taking into account the contribution of each label, or class, and then average. Since the threshold is 0.5 and the model will learn better the most frequent classes (this also due to adopting a loss function that doesn't really optimize the metric directly), what happens is that the majority of the classes receives low scores, not high enough to be greater than the decision threshold, which ultimately impacts the value of the metric. Therefore, we can see f2 scores curves that flat. It is more insightful to look at the loss curves. Models v10, v1.1 and v1.4 show that model is learning, whereas v1.2 and v1.3 s show very slow learning.

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "BASELINE MODELS LOSS AND MICRO F2-SCORE CURVES")
FIG : BASELINE MODELS LOSS AND MICRO F2-SCORE CURVES:  MICRO METRICS ARE COMPUTED TAKING INTO ACCOUNT THE CONTRIBUTION OF EACH LABEL, OR CLASS, AND THEN AVERAGE. SINCE THE THRESHOLD IS 0.5 AND THE MODEL WILL LEARN BETTER THE MOST FREQUENT CLASSES (THIS ALSO DUE TO ADOPTING A LOSS FUNCTION THAT DOESN'T REALLY OPTIMIZE THE METRIC DIRECTLY), WHAT HAPPENS IS THAT THE MAJORITY OF THE CLASSES RECEIVES LOW SCORES, NOT HIGH ENOUGH TO BE GREATER THAN THE DECISION THRESHOLD, WHICH ULTIMATELY IMPACTS THE VALUE OF THE METRIC. THEREFORE, WE CAN SEE F2 SCORES CURVES THAT FLAT.

I then tried several experiments with a pre-trained model, ResNet50. Afer a few iterations, the best optimization algorithm resulted to be Adam. The best performing loss function at the beginning (feature extractor mode) resulted to be soft f2 loss. However, when running in fine-tuning mode, especially increasing the number of unfrozen layers, binary crossentropy performed better. The strategy adopted with tuning the learning-rate was starting with 10ˆ-4 and using a Reduce LR callback with a decay to 10ˆ-6. 

The experiments using ResNet50 as a feature extractor, showed significant improvents w.r.t. the baseline models. However, even gradually increasing the training set (from 30% to 60%), the model didn't show improvements, plateauing around a loss value of  0.3 , without overfitting. I gradually unfroze top layers until overfitting, then I slowly increased the amount of data, and iteratated with these two parameters (number of layers to unfreeze and amount of data). Finally, I overfitted a fully trainable model, keeping, as recommended by Keras official guide, the model in inference mode (training=False) to avoid that the weights of batch normalized layers would be destroyed after the first iterations. At this point, the strategy to manage overfitting was first to do some data augmentation, then increasing the dropout rate.

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "RESNET50 AS FEATURE EXTRACTOR (ALL LAYERS FROZEN), TRAINED ON 30% OF THE ORIGINAL DATASET")
FIG : RESNET50 AS FEATURE EXTRACTOR (ALL LAYERS FROZEN), TRAINED ON 30% OF THE ORIGINAL DATASET;  LEFT COLUMN SHOWS PERFORMANCE OF THE MODEL TRAINED WITH BINARY CROSSENTROPY LOSS; RIGHT COLUMN SHOWS THE SAME MODEL TRAINED WITH SOFT-MICRO-F2 LOSS. THERE IS A SLIGHT INCREASE IN PERFORMANCE WITH SOFT-MICRO-F2 LOSS.

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "RESNET50 TRAINED ON 60% OF THE ORIGINAL DATASET, SOFT-MICRO-F2 LOSS")
FIG : RESNET50 TRAINED ON 60% OF THE ORIGINAL DATASET, SOFT-MICRO-F2 LOSS;  LEFT COLUMN SHOWS PERFORMANCE OF THE MODEL TRAINED WITH 20% OF THE TOP LAYERS TRAINABLE (FINE-TUNING); RIGHT COLUMN SHOWS THE SAME MODEL TRAINED AS FEATURE EXTRACTOR. THEY BOTH OVERFIT

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "RESNET50 TRAINED ON 100% OF THE ORIGINAL DATASET, BINARY CROSSENTROPY LOSS")
FIG : RESNET50 TRAINED ON 100% OF THE ORIGINAL DATASET, BINARY CROSSENTROPY LOSS;  MODEL TRAINED WITH 100% OF THE TOP LAYERS TRAINABLE (FINE-TUNING); IT CLEARLY OVERFITS

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "RESNET50 TRAINED WITH TF.DATA PIPELINE, ON 100% OF THE ORIGINAL DATASET, BINARY CROSSENTROPY LOSS")
FIG : RESNET50 TRAINED WITH TF.DATA PIPELINE, ON 100% OF THE ORIGINAL DATASET, BINARY CROSSENTROPY LOSS;  MODEL TRAINED WITH 100% OF THE TOP LAYERS TRAINABLE (FINE-TUNING); IT OVERFITS

This last configuration, data augmentation and L2 regularization, helped achievieng the best results obtained, that were around 3% less than state-of-the-art results. Below results on test set are summarised and displayed:

FIG : RESNET50 TRAINED ON 100% OF THE ORIGINAL DATASET, FINE-TUNED, WITH DATA AUGMENTATION AND L2 REGULARIZATION; MAX F2-SCORE PER LABEL

FIG : RESNET50 TRAINED ON 100% OF THE ORIGINAL DATASET, FINE-TUNED, WITH DATA AUGMENTATION AND L2 REGULARIZATION; THE MODEL ACHIEVES A 89.5% F2-SCORE ON TEST SET, WITHOUT OVERFITTING.

FIG : F2-SCORE FOR LABEL BY THRESHOLD; MODEL: RESNET50 TRAINED ON 100% OF THE ORIGINAL DATASET, FINE-TUNED, WITH DATA AUGMENTATION AND L2 REGULARIZATION. ON RARE LABELS RESULTS ARE SIGNIFICANTLY WORSE, ASLO DUE TO LACK OF SAMPLES IN THE TRAINING SET.

We posed three main questions we wanted to address when framing the business value of this project. 
- Can help tracking the exploitation of the Amazon basin in a more efficient way (i.e. less manual working)?

Yes, it can. Fine-tuning pre-trained models, as proven also by previous [works](https://arxiv.org/pdf/1703.00121.pdf),  allows to bypass the high costs of building a big datasets of satellite images. If we add the time-component, we can create a database of areas and monitor their evolution throughout time. Remote sensing data offers the chance to go well beyond scene classification: indeed we can add layers of information through spectral analysis (i.e. vegetation indices) or leveraging on SAR images. We can even think of adding other layers and sources of information, like sensors on the ground, creating a network of sensors.

- Where are the majority of fires located?

To answer to this question I merged different data sources and plotted them on a map (using Folium library). The heatmap shows fires and their radiative power, expressed in MW, which is the rate of emitted radiative energy by the fire at the time of the observation. It is possible to immediately identify an arch of fires, covering the sourthern part of the Amazon basin. In the northern state of Roraima, there is another cluster of fires, as well, it is possible to see a series of fires starting from the delta of the Amazon river, up until Manaus. 

It is significant to highlight how, confirming qualitative news available on media, that there is some significant overlapping between fires and mining sites in particular. Unfortunately, there was no data on farming activities, considering that they are ranked as one of the main causes of deforestation through fires.

- Can this tool be potentially used for verifying deforestation alerts?

Yes, it can. It could operate as an automated alert system as well as on request,  giving the coordinates of the area of the alert, the model would be able to classify the scene, fact-checking almost in real-time.

## Future works and conclusion

Due to time and computational constraints,  there are several ideas to improve the performance of the classifier that would help achieving more insightful recommendations:

1) Experimenting with full dataset and fine-tuning pretrained models, leveraging on known architectures such as DenseNet, Xception, and on parallel computing, possibly rewriting code using lower level Tensorflow (i.e. tape gradient, disabling eager mode, decorating functions with @tf.function)

2) Increase performance of classification leveraging on ensemble learning for CNNs, i.e.: decomposing the classification task in grouping labels into categories; designing models specialized in classifing each category; ensembling predictions via thresholds.

3) Adding time component (time-series analysis) to the geospatial analysis: being able to predict and assess scenes at different stages of their evolution (i.e. mining sites at early stages). This would requrie different architectures (i.e.  mixing CNNs with Recurrent Neural Networks).

4) Leveraging NIR channel for vegetation indices. I would also love to dig into the applications of deep learning to hyperspectral images.

Summarizing, this project demonstrates the capability of machine learning techniques  applied to computer vision, in particular to Remote Sensing images, as a building block for strategic decision making, such as the case of sustainable devolpment of a region. 
