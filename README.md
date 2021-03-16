# wikipedia summarizer transformer
the model is based of 'attention is all you need' paper, and trained in colab on the cnn_dailymail dataset, available to trax/tensorflow dataset.

# train the model
first you need to train your model, not pre-trained weight, and dataset are not added due to size limitation.
```cosole
user@name:~/summarizer$ python3 ./summarizer.py -t

```

## sample execution
first you need to
```cosole
user@name:~/summarizer$ python3 ./summarizer.py -w 'New York City'

```

## sample output
<details>
  <summary>
    New York City is the most densely populated major city in the United
    States. With almost 20 million people in its combined statistical
    area, it is one of the world's most populous megacities. New York City
    has been described as the cultural, financial, and media capital of
    the world.<EOS>The Duke of York was named in 1664 by the Duke of York City of NYC.
    The city was inhabited by Algonquian Native Americans. The city was
    inhabited by Algonquian Native Americans, including Lenape. The first
    non-native American inhabitant of what would become City was.<EOS>New Amsterdam launched its first modern fending off the coast of New
  </summary>
  New York City is the most densely populated major city in the United
States. With almost 20 million people in its combined statistical
area, it is one of the world's most populous megacities. New York City
has been described as the cultural, financial, and media capital of
the world.<EOS>The Duke of York was named in 1664 by the Duke of York City of NYC.
The city was inhabited by Algonquian Native Americans. The city was
inhabited by Algonquian Native Americans, including Lenape. The first
non-native American inhabitant of what would become City was.<EOS>New Amsterdam launched its first modern fending off the coast of New
York. The Dutch would eventually be deposed in the Glorious
Revolution. The Dutch would eventually be deposed in the Glorious
Revolution.<EOS>The Battle of Long Island was fought in August 1776 within the modern-
day borough of Brooklyn. The Battle of Long Island was fought in
August 1776 within the modern-day borough of Brooklyn. The British
Army and the First Division of the United States were all at Federal
Hall on Wall Street.<EOS>There was also extensive immigration from the German provinces.
Presence of immigrants was especially fierce in the U.S. in Sub-zero.
population.<EOS>The U.N. Headquarters was completed in 1952, solidifying New York's
global geopolitical influence. The massive event led to the ongoing
counter-racism protests in the early morning hours of June 28, 1969.
The New York City suffered the bulk of the economic damage and largest
loss of human life in the aftermath of the September 11, 2001
attacks.<EOS>The Hudson River flows through the Hudson River into New York City
from the U.S. state of New Jersey. The Bronx River flows through the
Hudson River into New York Bay. The Bronx River flows through the
Bronx and Westchester County. The Bronx River flows through the Bronx
and Westchester County, is the only freshwater river in the city's
land has been altered substantially by human intervention.<EOS>The Bronx is the largest central core city in the Outer Boroughs. It
is located in Queens of Brooklyn and the Bronx of Brooklyn. The Bronx
is the birthplace of hip hop music and culture.<EOS>Chrysler Building and Empire State Building (1957) are considered some
of the finest examples of the Art Deco style. The Condé Nast Building
(2000) is a prominent example of green design in American skyscrapers.
The Condé Nast Building (2000) is a prominent example of green design
in American skyscrapers.<EOS>The National Parks and the New York City of Garden Kingdom Island are
located in New York City. The Songwets are in the mood of the city and
the New York City of Garden. The Songwater Village in July 23, cites
the coldest the coldest month on record.<EOS>Historic sites under federal management on Manhattan Island include
Castle Clinton National Monument. The Chapel is the historic river
where the historic Landmark is located in Greenwich Village. The
National Monument is located in New York City and the Central Park is
the most visited urban park in the United States.<EOS>More than twice as many people live in New York City as compared to
Los Angeles, the second-most populous U.S. city. City gained more
residents between April 2010 and July 2014 (316,000) than any other
U.S. city.<EOS>Koreans made up 1.2% of New York City's population in India and Nepal.
Koreans made up 1.2% of the city's population in 2010. Koreans made up
1.2% of New York's population in 2010. Koreans made up 1.2% of New
York's population, with Bangladeshis and Pakistanis.<EOS>The annual New York City Pride March (or gay pride parade) was the
literal gay metropolis for hundreds of thousands of immigrants. The
annual Queens Pride Parade is held in Jackson Heights and is
accompanied by the ensuing Multicultural armies. Gay Pride parade is
held in Jackson Heights and is accompanied by the ensuing
Multicultural sensitivity.<EOS>The American Orthodox Catholic Church (mainstream and independent)
were the largest Christian groups. The American Orthodox Christian
Church (mainstream and independent) were the largest Christian
groups.<EOS>Entrepreneurs were forming a "Chocolate District" in Brooklyn as of
2014. Entrepreneurs were forming a "Chocolate District" in Brooklyn as
of 2014. One of the world's largest chocolatiers, continues to be
headquartered in Manhattan.<EOS>Accelerator, a biotechnology startup, had raised more than $30 million
from investors. The New York City of New York City is home to some of
the nation's highest-rated market rents in the U.S. State Department
of Technology.<EOS>The U.N. office of the Statue of Liberty and Ellis Island is the
largest in North America. The New York City is also a center for the
advertising, music, newspaper, digital media and digital media.<EOS>The Village Voice, historically the largest alternative newspaper in
the United States, announced in 2017 that it would cease publication
of its print edition and convert to digital venture.<EOS>Focus of Memorial Sloan, Rockefeller University, SUNY Downstate
Medical Center, Albert Einstein College of Medicine,. and Weill
Cornell Medical College of Medicine,. Harvard-Israel Institute of
Technology venture on Roosevelt Island. The graduates of SUNY Maritime
College in the Bronx earned the highest average annual salary of any
university graduates in the U.S, $144,000 as of 2017.<EOS>The FDNY is the largest municipal fire department in the U.S. and the
second largest in the world after the Tokyo Fire Department. The FDNY
has been described as the cultural capital of the world by New York's
Baruch College.<EOS>The dedication of the arts and cultural landmarks are held in New York
City. The dedication of the landmarks are ranged from the Empire to
the theatre of the Walking Arts.<EOS>The city is home to "nearly one thousand of the finest and most
diverse haute cuisine restaurants in the world" . There are 27,043
restaurants in the city, up from 24,865 in 2017. The Queens Night
Market in Flushing Meadows attracts more than 10 thousand people
nightly to sample food from more than 85 countries.<EOS>The Yankee Stadium and Ebbets are located in New York City. Madison
Square Garden, its predecessor, the Yankee Stadium and Ebbets Field
are sporting venues located in New York City. The New York City has
been described as the 'Capital of Baseball' and the Brooklyn Dodgers
have won the World Series twice.<EOS>The annual United States Open Tennis Championships is held in New
York. The annual United States Open Championship is held at the
National Tennis Center in Flushing Meadows-Corona Park, Queens. The
Millrose Games is the best of the city in the United States.<EOS>The City Water Tunnel No. 3 is the largest capital construction
project in the city's history. The city's largest capital construction
project in the city's history. The city's mayor and council members
are elected to four-year terms.<EOS>The New York City has not been carried by a Republican in a statewide
or presidential election since Calvin Coolidge won the five boroughs
in 1924. City has a strong imbalance of payments with the national and
state governments. City residents and businesses also sent an
additional $4.1 billion in the 2009 fiscal year to the state of New
York than the city received in return.<EOS>Three major rail trains are in New York and New Jersey City. The Port
Authority Rail trains are dedicated to Staten Island, New York and New
Jersey.<EOS>The Manhattan Bridge is the longest suspension bridge in the world
from its opening until 1903. The New York City is also known for its
rules regarding turning at red lights.<EOS>
</details>

# dataset installation
make sure to install [cnn_dailymail](https://www.tensorflow.org/datasets/catalog/cnn_dailymail) under subdirectory data
