getwd()
setwd('D:/Udacity/DAND/p4')
data("diamonds")

library(ggplot2)

ggplot(aes(x = price, fill=cut), data = diamonds) + geom_histogram() +
  facet_wrap(~ color) +
  scale_x_log10() +
  scale_fill_brewer(type = 'qual')

ggplot(data = diamonds, aes(x = table, y = price)) +
  geom_jitter(alpha = 1,  aes(color = cut)) +
  scale_color_brewer(type = 'qual')+
  coord_cartesian(xlim = c(50,65))
  scale_x_continuous(breaks = seq(50,65, 1))
??breaks
  
  
diamonds$volume <- diamonds$x*diamonds$y*diamonds$z

ggplot(aes(x = volume, y = price), data = diamonds) +
  geom_jitter(aes(color = clarity))+
  scale_y_log10()+
  coord_cartesian(xlim = c(0,300)) +
  scale_x_continuous(breaks = seq(0, 300,100))+
  scale_color_brewer(type = 'div')

pf <- read.csv('pseudo_facebook.tsv', sep = '\t')
pf <- transform(pf, prop_initiated = ifelse(friend_count == 0, 0, friendships_initiated/friend_count))
pf$year_joined <- floor(2014 - pf$tenure / 365)
pf$year_joined.bucket <- cut(pf$year_joined, c(2004,2009,2011,2012, 2014))
ggplot(data = subset(pf, !is.na(prop_initiated) & !is.na(year_joined.bucket)),
       aes(x = tenure, y = prop_initiated)) +
  geom_line(aes(color = year_joined.bucket), stat = "summary", fun.y = median)+
  geom_smooth(method = 'gam')

summary(subset(pf, year_joined.bucket =='(2012,2014]')$prop_initiated)

ggplot(aes(x = cut, y = price / carat), data = diamonds) +
  geom_jitter(aes(color = color),alpha = 1)+
  facet_grid(~ clarity)+
  scale_color_brewer(type = 'div')


install.packages('tidyverse')
library(tidyverse)
library(readxl)
library(countrycode)

years_women <- read_excel("./data/Years in school women 25-34.xlsx")
years_women <- rename(years_women,Country = `Row Labels`) ## rename column
years_women[29,1] <- "Central African Republic" ## changed unofficial english country name from 'Central African Rep.' to 'Central African Republic'
years_women_region <- mutate(years_women, 
                             regions = countrycode(years_women[[1]], 'country.name', 'region')) ## add column to indicate region of each country
##years_women_region$regions <- factor(years_men_region$regions) uncomment to "regions" column to a factor

## The line below reorders the columns, resulting in 'Country', 'regions' and the rest
years_women_region <- years_women_region %>% select(Country, regions, everything())
## Set the years as observations to create a tidy data frame. 
tidyDF <- gather(years_women_region, "year", "n", 3:42)

## filter() is used like subset() to limit the number of 'regions'
ggplot( filter(tidyDF, regions == 'Caribbean' | regions == "Western Asia"),
        aes(x=n, y=year, color=Country, group=Country)) +
  geom_line(show.legend = F) +
  facet_wrap(~regions, nrow = 2) +
  directlabels::geom_dl(aes(label = Country), method = "smart.grid") +
  scale_fill_brewer(palette = "Spectral")

ggsave("c_wa.jpeg")

  
  
