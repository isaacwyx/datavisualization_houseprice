# Exploratory Data Analysis (EDA) on Houses Prices in Ames, Iowa
## by Yuxiang Wu


## Dataset

The purpose of this project is to practise my dataset exploration skills by using Python. Thus, I will perform an exploratory data analysis (EDA) on a dataset that contains information of 1460 houses with 81 variables on each house, including neighbourhoods, living areas, building types, lot size, and many others. The original dataset was downloaded from a Kaggle competition, which was to challenge participants to predict the final price of each home in Ames, Iowa. The complete downloaded file contains two sets of data for testing and training, respectively. The dataset I will be working on is the one for training, as it contains the sale price for each house.
<br>
To do preliminary wrangling, I will inspect the data structure, select the variables in the dataset that will help support the investigation, and create new variables if necessary.


## Summary of Findings

#### Finding 1
The average price of a house is approximately 180,000, and the median price is slightly less than 200,000. 
<br>
There are multiple relatively expensive houses in the dataset, and the most expensive one is over 700,000. However, I do not consider those costly houses as outliers because each higher price range has a few data points, and they are not extremely far away from the majority of the data.

#### Finding 2
The majority of the houses in Ames have lot areas in between 5,000 and 15,000 square feet, total living areas in between 600 and 2,250 square feet, and basement areas in between 500 and 2,000 square feet.

#### Finding 3
The average house is taking slightly over 10% of its lot area, and the median building_coverage ratio is approximately 13%. There are a few houses that have larger building_coverage ratio, and some houses take up almost half of their lot size. 

#### Finding 4
Despite 15.7% of the house listings that do not specify their neighbourhoods, the three neighbourhoods that have the most house information in the dataset are CollgCr (Clear Creek), Old Town, and Somerst (Somerset).

##### Finding 5
- The type of dwelling for most of the houses in the dataset is 1Fam (Single-family Detached)
- Most of the houses in the dataset has 1 or 2 full bathrooms
- The majority of the houses have 3 bedrooms, and a considerable amount of houses have 2 or 4 bedrooms
- The most popular garage types are Attchd (Attached to home) or Detchd (Detached from home)
- Most of the houses receive a house condition rating of 5, and fewer houses have higher ratings
- Most of the houses in the dataset are in the RL (Residential Low Density) zone, and a few houses are in the RM (Residential Medium Density) zone.

#### Finding 6
There is a positive correlation between the sale price and total living area, and most houses have listing prices under 300,000 with a total living area of less than 2500 square feet. There is also a positive relationship between the sale price and basement area, and the basement areas of most houses in the City of Ames are between 500 to 2000 square feet.

#### Finding 7
In general, there is a negative correlation between lot area and building_coverage ratio. From the above areas histogram, it can be seen that the sizes of houses do not vary as much as the sizes of lots. Houses with a larger building coverage ratio usually have a smaller lot. However, lots of the data points stacked together in a concentrated area.

#### Finding 8
Apperantly, a house with more full bathrooms comes with a higher price tag. On the other hand, the relation between the sale price and bedrooms is somewhat polynomial, which is not in my expectation.

#### Finding 9
Interestingly, sale price does not associate with house condition ratings. Instead, the three neighbourhoods with the highest sale prices only have an average house condition rating. The three most expensive neighbourhoods are NoRidge, NridgHt, and StoneBr. The three neighbourhoods with the highest house condition ratings are Crawfor, OldTown, and Veenker. Next, I will take a deeper look at the houses in the three most expensive neighbourhoods.

#### Finding 10
- Most of the houses in the three most expensive neighbourhoods are Single-family Detached, and the rest of dwellings in these neighbourhoods are either TwnhsE (Townhouse End Unit) or Twnhs (Townhouse Inside Unit)
- Most of the houses in the three most expensive neighbourhoods have attached garage, and the others have a built-in garage.
- In NoRidge, most families have 4 bedrooms. In NridgHt, most families have 3 bedrooms. In StoneBr, most families have 2 bedrooms.
- Most of the houses in the three most expensive neighbourhoods have 2 full bathrooms above ground.
- Most of the houses in the three most expensive neighbourhoods are in a RL (Residential Low Density) area, while the others are in a RM (Residential Medium Density) area.

#### Finding 11
- The most popular dwelling type in the city of Ames is Single-family Detached. Houses with price tags over $300,000 are mostly Single-family Detached.
- The Two-family Conversion type of house disappeared from the market after 1970.
- Duplex became less popular after 1980
- In general, Townhouse End Units are more expensive than Townhouse Inside Units

#### Finding 12 
- Houses located in the residential medium density area are usually sold less expensive than other areas, except for those located in a commercial area.
- Most of the houses in the city of Ames are sold below 300,000 and have a building coverage ratio of less than 0.2
- Most of the houses are located in a residential low-density area.
- Houses worth more than $300,000 are more likely located in a residential low-density area.


## Key Insights for Presentation

#### House price and house areas
>There is a positive correlation between the sale price and total living area, and most houses have listing prices under 300,000 with a total living area of less than 2500 square feet. There is also a positive relationship between the sale price and basement area, and the basement areas of most houses in the City of Ames are between 500 to 2000 square feet.

#### Sales, House Condition, and Neighborhood
>Interestingly, sale price does not associate with house condition ratings. Instead, the three neighbourhoods with the highest sale prices only have an average house condition rating. The three most expensive communities are NoRidge, NridgHt, and StoneBr. The three neighbourhoods with the highest house condition ratings are Crawfor, OldTown, and Veenker.  

#### Popular House Types Over Years
>The most popular dwelling type in the city of Ames is Single-family Detached. Houses with price tags over $300,000 are mostly Single-family Detached. The Two-family Conversion type of house disappeared from the market after 1970. Duplex became less popular after 1980. In general, Townhouse End Units are more expensive than Townhouse Inside Units


