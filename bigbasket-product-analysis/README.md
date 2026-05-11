# BigBasket Product Analysis

Exploratory analysis of the product catalogue from BigBasket, an Indian online grocery and e-commerce platform.

## Dataset

~28,000 product records. Source: [Kaggle – BigBasket Entire Product List](https://www.kaggle.com/datasets/surajjha101/bigbasket-entire-product-list-28k-datapoints).

Key columns: `category`, `sub_category`, `brand`, `type`, `market_price`, `sale_price`, `rating`.

A `discount` column is derived as `(market_price - sale_price) / market_price`.

## Analysis highlights

- Category share breakdown
- Rating distribution (~31% of ratings are missing)
- Price and discount patterns across categories and brands

## Tech stack

R · tidyverse · Cairo

## Report

https://tokarskipatryk.github.io/data-analysis/bigbasket-product-analysis/big-basket.html
