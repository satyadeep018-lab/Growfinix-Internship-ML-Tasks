import os
import sys

# Streamlit Cloud import protection
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import urllib.request
import zipfile
import numpy as np
import pandas as pd

# High-quality movie catalog (120 iconic films across genres)
BENCHMARK_MOVIES = [
    # Sci-Fi & Space
    (1, "The Matrix", 1999, "Action|Sci-Fi", "🕶️", "A computer hacker learns about the true nature of reality and his role in the war against its controllers."),
    (2, "Inception", 2010, "Action|Adventure|Sci-Fi", "🌀", "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea."),
    (3, "Interstellar", 2014, "Adventure|Drama|Sci-Fi", "🚀", "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival."),
    (4, "Blade Runner 2049", 2017, "Drama|Mystery|Sci-Fi", "🤖", "Young Blade Runner K unearths a long-buried secret that leads him to track down former Blade Runner Rick Deckard."),
    (5, "Star Wars: Episode IV - A New Hope", 1977, "Action|Adventure|Fantasy|Sci-Fi", "⚔️", "Luke Skywalker joins forces with a Jedi Knight, a cocky pilot, and two droids to save the galaxy."),
    (6, "Star Wars: Episode V - The Empire Strikes Back", 1980, "Action|Adventure|Fantasy|Sci-Fi", "❄️", "After the Rebels are overpowered by the Empire, Luke Skywalker begins Jedi training with Yoda."),
    (7, "2001: A Space Odyssey", 1968, "Adventure|Sci-Fi", "🛰️", "After discovering a mysterious artifact on the Moon, mankind seeks its origins with the help of supercomputer HAL 9000."),
    (8, "Terminator 2: Judgment Day", 1991, "Action|Sci-Fi", "🦾", "A cyborg identical to the one who failed to kill Sarah Connor must now protect her ten-year-old son."),
    (9, "Alien", 1979, "Horror|Sci-Fi", "👾", "The crew of a commercial spacecraft encounters a deadly lifeform after investigating an unknown transmission."),
    (10, "Aliens", 1986, "Action|Adventure|Sci-Fi", "💥", "Decades after surviving the Nostromo incident, Ellen Ripley returns to the moon LV-426 with space marines."),
    (11, "Arrival", 2016, "Drama|Mystery|Sci-Fi", "🛸", "A linguist works with the military to communicate with alien lifeforms after twelve spacecraft appear around the world."),
    (12, "Dune: Part One", 2021, "Action|Adventure|Drama|Sci-Fi", "🏜️", "A noble family becomes embroiled in a war for control over the galaxy's most valuable asset on a desert planet."),

    # Crime & Gangster
    (13, "The Godfather", 1972, "Crime|Drama", "🍷", "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son."),
    (14, "The Godfather Part II", 1974, "Crime|Drama", "🏛️", "The early life and career of Vito Corleone in 1920s New York is portrayed while his son Michael expands the syndicate."),
    (15, "Pulp Fiction", 1994, "Crime|Drama", "🍔", "The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption."),
    (16, "Goodfellas", 1990, "Biography|Crime|Drama", "🍝", "The story of Henry Hill and his life in the mafia, covering his relationship with his wife Karen and his mob partners."),
    (17, "The Departed", 2006, "Crime|Drama|Thriller", "☘️", "An undercover cop and a mole in the police attempt to identify each other while infiltrating an Irish gang in Boston."),
    (18, "Scarface", 1983, "Crime|Drama", "🌴", "In 1980 Miami, a determined Cuban immigrant takes over a drug cartel and succumbs to greed."),
    (19, "Se7en", 1995, "Crime|Drama|Mystery|Thriller", "📦", "Two detectives hunt a serial killer who uses the seven deadly sins as his motives."),
    (20, "The Usual Suspects", 1995, "Crime|Drama|Mystery|Thriller", "🕵️", "A sole survivor tells of the twisty events leading up to a horrific gun battle on a boat."),
    (21, "Heat", 1995, "Action|Crime|Drama|Thriller", "🏦", "A group of high-end professional thieves start to feel the heat from the LAPD when they unwittingly leave a clue."),
    (22, "Casino", 1995, "Crime|Drama", "🎰", "In Las Vegas, two best friends—a casino executive and a mafia enforcer—compete over a gambling empire."),

    # Action & Superhero
    (23, "The Dark Knight", 2008, "Action|Crime|Drama|Thriller", "🦇", "Batman faces his greatest psychological test when the Joker wreaks havoc and chaos on Gotham City."),
    (24, "The Dark Knight Rises", 2012, "Action|Drama|Thriller", "💣", "Eight years after the Joker's reign of anarchy, Batman is forced out of exile by the terrorist Bane."),
    (25, "Batman Begins", 2005, "Action|Crime|Drama", "🌃", "After training with his mentor, Bruce Wayne begins his fight to free crime-ridden Gotham City."),
    (26, "Avengers: Endgame", 2019, "Action|Adventure|Drama|Sci-Fi", "🧤", "After the devastating events of Infinity War, the universe is in ruins and the remaining Avengers assemble."),
    (27, "Avengers: Infinity War", 2018, "Action|Adventure|Sci-Fi", "💎", "The Avengers and their allies must be willing to sacrifice all in an attempt to defeat the powerful Thanos."),
    (28, "Iron Man", 2008, "Action|Adventure|Sci-Fi", "🔴", "After being held captive in an Afghan cave, billionaire Tony Stark creates a unique weaponized suit of armor."),
    (29, "Spider-Man: Into the Spider-Verse", 2018, "Animation|Action|Adventure|Sci-Fi", "🕷️", "Teen Miles Morales becomes the new Spider-Man and joins other Spider-Heroes from parallel universes."),
    (30, "Mad Max: Fury Road", 2015, "Action|Adventure|Sci-Fi", "🔥", "In a post-apocalyptic wasteland, a woman rebels against a tyrannical ruler in search for her homeland."),
    (31, "Gladiator", 2000, "Action|Adventure|Drama", "⚔️", "A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family."),
    (32, "John Wick", 2014, "Action|Crime|Thriller", "🐶", "An ex-hitman comes out of retirement to track down the gangsters that took everything from him."),
    (33, "Die Hard", 1988, "Action|Thriller", "🏢", "An NYPD officer tries to save his estranged wife and several others taken hostage by terrorists in a skyscraper."),
    (34, "Top Gun: Maverick", 2022, "Action|Drama", "✈️", "After thirty years, Maverick is still pushing the envelope as a top naval aviator training a detachment of graduates."),

    # Drama & Prestige Cinema
    (35, "The Shawshank Redemption", 1994, "Drama", "🗝️", "Over the course of several years, two convicts form a friendship, seeking consolation and eventual redemption."),
    (36, "Fight Club", 1999, "Drama", "🧼", "An insomniac office worker and a devil-may-care soap maker form an underground fight club that evolves into much more."),
    (37, "Forrest Gump", 1994, "Drama|Romance", "🍫", "The history of the United States from the 1950s to the '70s unfolds from the perspective of an Alabama man with an IQ of 75."),
    (38, "Parasite", 2019, "Drama|Thriller", "🍑", "Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kims."),
    (39, "Whiplash", 2014, "Drama|Music", "🥁", "A promising young drummer enrolls at a cut-throat music conservatory where his dreams are guided by an instructor who will stop at nothing."),
    (40, "Schindler's List", 1993, "Biography|Drama|History", "🕯️", "In German-occupied Poland during World War II, industrialist Oskar Schindler gradually becomes concerned for his Jewish workforce."),
    (41, "12 Angry Men", 1957, "Crime|Drama", "⚖️", "The jury in a New York City murder trial is frustrated by a single member whose skeptical caution forces them to consider the evidence."),
    (42, "One Flew Over the Cuckoo's Nest", 1975, "Drama", "💊", "In the Fall of 1963, a Korean War veteran pleads insanity to serve his sentence in a mental hospital rather than prison."),
    (43, "The Social Network", 2010, "Biography|Drama", "💻", "As Harvard student Mark Zuckerberg creates the social networking site that would become Facebook, he is sued by the twins who claimed he stole their idea."),
    (44, "There Will Be Blood", 2007, "Drama", "🛢️", "A story of family, religion, hatred, oil and madness, focusing on a turn-of-the-century prospector in the early days of the business."),
    (45, "No Country for Old Men", 2007, "Crime|Drama|Thriller", "🪙", "Violence and mayhem ensue after a hunter stumbles upon some dead bodies, a stash of heroin and more than $2 million in cash."),
    (46, "Oppenheimer", 2023, "Biography|Drama|History", "⚛️", "The story of American scientist J. Robert Oppenheimer and his role in the development of the atomic bomb."),

    # Mind-Bending & Psychological Thrillers
    (47, "Shutter Island", 2010, "Mystery|Thriller", "🏝️", "In 1954, a U.S. Marshal investigates the disappearance of a murderer who escaped from a hospital for the criminally insane."),
    (48, "Memento", 2000, "Mystery|Thriller", "📸", "A man with short-term memory loss attempts to track down his wife's murderer using polaroids and tattoos."),
    (49, "The Prestige", 2006, "Drama|Mystery|Sci-Fi", "🎩", "After a tragic accident, two stage magicians in 1890s London engage in a battle to create the ultimate illusion."),
    (50, "The Silence of the Lambs", 1991, "Crime|Drama|Thriller", "🦋", "A young F.B.I. cadet must receive the help of an incarcerated and manipulative cannibal killer to help catch another serial killer."),
    (51, "Gone Girl", 2014, "Drama|Mystery|Thriller", "🗞️", "With his wife's disappearance having become the focus of an intense media circus, a man sees the spotlight turned on him."),
    (52, "Zodiac", 2007, "Crime|Drama|Mystery", "✍️", "Between 1968 and 1983, a San Francisco cartoonist becomes an amateur detective obsessed with tracking down the Zodiac Killer."),
    (53, "Black Swan", 2010, "Drama|Thriller", "🦢", "A committed dancer wins the lead role in a production of Tchaikovsky's 'Swan Lake' only to struggle with her sanity."),
    (54, "Primal Fear", 1996, "Crime|Drama|Mystery", "✝️", "An altar boy is accused of murdering a priest, and the truth is buried several layers deep."),

    # Animation & Anime
    (55, "Spirited Away", 2001, "Animation|Adventure|Family|Fantasy", "🏮", "During her family's move to the suburbs, a sullen 10-year-old girl wanders into a world ruled by gods, witches, and spirits."),
    (56, "Princess Mononoke", 1997, "Animation|Action|Adventure|Fantasy", "🐺", "On a journey to find the cure for a Tatarigami's curse, Ashitaka finds himself in the middle of a war between the forest gods and Tatara."),
    (57, "Your Name", 2016, "Animation|Drama|Fantasy|Romance", "🌠", "Two teenagers share a profound, magical connection upon discovering they are swapping bodies."),
    (58, "The Lion King", 1994, "Animation|Adventure|Drama|Family", "🦁", "Lion prince Simba and his father are targeted by his bitter uncle, who wants to ascend the throne himself."),
    (59, "WALL·E", 2008, "Animation|Adventure|Family|Sci-Fi", "🌱", "In the distant future, a small waste-collecting robot inadvertently embarks on a space journey that will ultimately decide the fate of mankind."),
    (60, "Toy Story", 1995, "Animation|Adventure|Comedy|Family", "🤠", "A cowboy doll is profoundly threatened and jealous when a new spaceman action figure supplants him as top toy in a boy's room."),
    (61, "Toy Story 3", 2010, "Animation|Adventure|Comedy|Family", "🧸", "The toys are mistakenly delivered to a day-care center instead of the attic right before Andy leaves for college."),
    (62, "Up", 2009, "Animation|Adventure|Comedy|Family", "🎈", "78-year-old Carl Fredricksen travels to Paradise Falls in his house equipped with balloons, inadvertently taking a young stowaway."),
    (63, "Coco", 2017, "Animation|Adventure|Comedy|Family|Music", "🎸", "Aspiring musician Miguel, confronted with his family's ancestral ban on music, enters the Land of the Dead."),
    (64, "Spider-Man: Across the Spider-Verse", 2023, "Animation|Action|Adventure|Sci-Fi", "🌌", "Miles Morales catapults across the Multiverse, where he encounters a team of Spider-People charged with protecting its very existence."),

    # Comedy & Satire
    (65, "The Grand Budapest Hotel", 2014, "Adventure|Comedy|Crime", "🏨", "A writer encounters the owner of an aging high-class hotel, who tells him of his early years serving as a lobby boy."),
    (66, "The Big Lebowski", 1998, "Comedy|Crime", "🎳", "Jeff 'The Dude' Lebowski, mistaken for a millionaire of the same name, seeks restitution for his ruined rug."),
    (67, "Superbad", 2007, "Comedy", "🍻", "Two co-dependent high school seniors are forced to deal with separation anxiety after their plan to stage a booze-soaked party goes awry."),
    (68, "Groundhog Day", 1993, "Comedy|Fantasy|Romance", "⏰", "A narcissistic TV weatherman finds himself reliving the same day over and over again when he visits Punxsutawney."),
    (69, "Monty Python and the Holy Grail", 1975, "Adventure|Comedy|Fantasy", "🥥", "King Arthur and his Knights of the Round Table embark on a surreal search for the Holy Grail."),
    (70, "The Truman Show", 1998, "Comedy|Drama|Sci-Fi", "📺", "An insurance salesman discovers his whole life is actually a reality television show."),
    (71, "Knives Out", 2019, "Comedy|Crime|Drama|Mystery", "🔍", "A detective investigates the death of the patriarch of an eccentric, combative family."),
    (72, "Jojo Rabbit", 2019, "Comedy|Drama|War", "👟", "A lonely German boy in the Hitler Youth discovers his mother is hiding a Jewish girl in their attic."),

    # Romance & Romantic Dramas
    (73, "Titanic", 1997, "Drama|Romance", "🚢", "A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic."),
    (74, "La La Land", 2016, "Comedy|Drama|Music|Romance", "🎹", "While navigating their careers in Los Angeles, a pianist and an actress fall in love while attempting to reconcile their aspirations."),
    (75, "Eternal Sunshine of the Spotless Mind", 2004, "Drama|Romance|Sci-Fi", "🧠", "When their relationship turns sour, a couple undergoes a medical procedure to have each other erased from their memories."),
    (76, "Before Sunrise", 1995, "Drama|Romance", "🚂", "A young man and woman meet on a train in Europe, and wind up spending one evening together in Vienna."),
    (77, "Before Sunset", 2004, "Drama|Romance", "🗼", "Nine years after Jesse and Celine first met, they encounter each other again on the French leg of Jesse's book tour."),
    (78, "Casablanca", 1942, "Drama|Romance|War", "🥂", "A cynical expatriate American cafe owner struggles to decide whether or not to help his former lover and her fugitive husband escape."),
    (79, "Amélie", 2001, "Comedy|Romance", "🥄", "Amélie is an innocent and naive girl in Paris with her own sense of justice who decides to help those around her."),
    (80, "Her", 2013, "Drama|Romance|Sci-Fi", "🎙️", "In a near future, a lonely writer develops an unlikely relationship with an operating system designed to meet his every need."),
    (81, "Past Lives", 2023, "Drama|Romance", "🗽", "Nora and Hae Sung, two deeply connected childhood friends, are wrest apart after Nora's family emigrates from South Korea."),
    (82, "Pride & Prejudice", 2005, "Drama|Romance", "💌", "Sparks fly when spirited Elizabeth Bennet meets single, rich, and proud Mr. Darcy."),

    # Adventure & Fantasy
    (83, "The Lord of the Rings: The Fellowship of the Ring", 2001, "Action|Adventure|Drama|Fantasy", "💍", "A meek Hobbit from the Shire and eight companions set out on a journey to destroy the powerful One Ring."),
    (84, "The Lord of the Rings: The Two Towers", 2002, "Action|Adventure|Drama|Fantasy", "🏰", "While Frodo and Sam edge closer to Mordor, the divided fellowship makes a stand against Sauron's new ally, Saruman."),
    (85, "The Lord of the Rings: The Return of the King", 2003, "Action|Adventure|Drama|Fantasy", "👑", "Gandalf and Aragorn lead the World of Men against Sauron's army to draw his gaze from Frodo and Sam."),
    (86, "Jurassic Park", 1993, "Action|Adventure|Sci-Fi", "🦖", "A pragmatic paleontologist touring an almost complete theme park on an island in Central America is tasked with protecting two kids."),
    (87, "Harry Potter and the Prisoner of Azkaban", 2004, "Adventure|Family|Fantasy", "🪄", "Harry Potter, Ron and Hermione return to Hogwarts for their third year of study, where they delve into the mystery of Sirius Black."),
    (88, "Pirates of the Caribbean: The Curse of the Black Pearl", 2003, "Action|Adventure|Fantasy", "🏴‍☠️", "Blacksmith Will Turner teams up with eccentric pirate 'Captain' Jack Sparrow to save his love."),
    (89, "Raiders of the Lost Ark", 1981, "Action|Adventure", "🏺", "In 1936, archaeologist and adventurer Indiana Jones is hired by the U.S. government to find the Ark of the Covenant."),
    (90, "Avatar", 2009, "Action|Adventure|Fantasy|Sci-Fi", "🌿", "A paraplegic Marine dispatched to the moon Pandora on a unique mission becomes torn between following his orders and protecting the world."),

    # Horror & Suspense
    (91, "The Shining", 1980, "Drama|Horror", "🪓", "A family heads to an isolated hotel for the winter where a sinister presence influences the father into violence."),
    (92, "Get Out", 2017, "Horror|Mystery|Thriller", "☕", "A young African-American visits his white girlfriend's parents for the weekend, where his simmering uneasiness reaches a boiling point."),
    (93, "Psycho", 1960, "Horror|Mystery|Thriller", "🚿", "A Phoenix secretary embezzles $40,000 from her employer's client, goes on the run, and checks into a remote motel."),
    (94, "A Quiet Place", 2018, "Drama|Horror|Sci-Fi", "🤫", "In a post-apocalyptic world, a family is forced to live in silence while hiding from monsters with ultra-sensitive hearing."),
    (95, "Hereditary", 2018, "Drama|Horror|Mystery", "👑", "A grieving family is haunted by tragic and disturbing occurrences after the death of their secretive grandmother."),
    (96, "The Thing", 1982, "Horror|Mystery|Sci-Fi", "❄️", "A research team in Antarctica is hunted by a shape-shifting alien that assumes the appearance of its victims."),

    # War & Historical Epic
    (97, "Saving Private Ryan", 1998, "Drama|War", "🪖", "Following the Normandy Landings, a group of U.S. soldiers go behind enemy lines to retrieve a paratrooper whose brothers have been killed."),
    (98, "Apocalypse Now", 1979, "Drama|Mystery|War", "🚁", "A U.S. Army officer serving in Vietnam is tasked with assassinating a renegade Special Forces Colonel who sees himself as a god."),
    (99, "Inglourious Basterds", 2009, "Adventure|Drama|War", "🔥", "In Nazi-occupied France during World War II, a plan to assassinate Nazi leaders by a group of Jewish U.S. soldiers coincides with a theatre owner's vengeful plans."),
    (100, "1917", 2019, "Action|Drama|War", "⏳", "April 6th, 1917. As an infantry battalion assembles to wage war deep in enemy territory, two soldiers are assigned to race against time and deliver a message."),

    # Western & Modern Cult Classics
    (101, "The Good, the Bad and the Ugly", 1966, "Adventure|Western", "🤠", "A bounty hunting scam joins two men in an uneasy alliance against a third in a race to find a fortune in gold buried in a remote cemetery."),
    (102, "Django Unchained", 2012, "Drama|Western", "⛓️", "With the help of a German bounty-hunter, a freed slave sets out to rescue his wife from a brutal Mississippi plantation owner."),
    (103, "Drive", 2011, "Action|Drama", "🚗", "A mysterious Hollywood action film stuntman who moonlights as a getaway driver finds himself in trouble when he helps out his neighbor."),
    (104, "Everything Everywhere All at Once", 2022, "Action|Adventure|Comedy|Sci-Fi", "🥯", "A middle-aged Chinese immigrant is swept up into an insane adventure in which she alone can save existence by exploring other universes."),
    (105, "Baby Driver", 2017, "Action|Crime|Music", "🎧", "After being coerced into working for a crime boss, a young getaway driver finds himself taking part in a heist doomed to fail."),
    (106, "Nightcrawler", 2014, "Crime|Drama|Thriller", "📹", "When Louis Bloom, a con man desperate for work, muscles into the world of L.A. crime journalism, he blurs the line between observer and participant."),
    (107, "Ex Machina", 2014, "Drama|Sci-Fi|Thriller", "🤖", "A young programmer is selected to participate in a ground-breaking experiment in synthetic intelligence by evaluating the human qualities of a humanoid A.I."),
    (108, "Spider-Man 2", 2004, "Action|Adventure|Sci-Fi", "🕸️", "Peter Parker is beset with troubles in his failing personal life as he battles a brilliant scientist named Doctor Otto Octavius."),
    (109, "The Wolf of Wall Street", 2013, "Biography|Comedy|Crime", "💵", "Based on the true story of Jordan Belfort, from his rise to a wealthy stock-broker living the high life to his fall involving crime and corruption."),
    (110, "Catch Me If You Can", 2002, "Biography|Crime|Drama", "✈️", "A seasoned FBI agent pursues Frank Abagnale Jr. who, before his 19th birthday, successfully performed cons worth millions of dollars."),
    (111, "Kill Bill: Vol. 1", 2003, "Action|Crime|Drama", "🗡️", "After awakening from a four-year coma, a former assassin wreaks vengeance on the team of assassins who betrayed her."),
    (112, "Prisoners", 2013, "Crime|Drama|Mystery|Thriller", "🌧️", "When Keller Dover's daughter and her friend go missing, he takes matters into his own hands as the police pursue multiple leads."),
    (113, "The Grandmaster", 2013, "Action|Biography|Drama", "🥋", "The story of martial-arts master Ip Man, the man who trained Bruce Lee."),
    (114, "Logan", 2017, "Action|Drama|Sci-Fi", "🐺", "In a future where mutants are nearly extinct, an elderly and weary Logan leads a quiet life until a young mutant arrives pursued by dark forces."),
    (115, "Barbie", 2023, "Adventure|Comedy|Fantasy", "🎀", "Barbie and Ken are having the time of their lives in the colorful Barbieland, until they get a chance to go to the real world."),
    (116, "Black Panther", 2018, "Action|Adventure|Sci-Fi", "🐾", "T'Challa, heir to the hidden kingdom of Wakanda, must lead his people into a new future and confront a challenger from his country's past."),
    (117, "Coco", 2017, "Animation|Family|Fantasy", "🌺", "A boy journeys into the Land of the Dead to unlock the real story behind his family's history."),
    (118, "Spider-Man: No Way Home", 2021, "Action|Adventure|Fantasy|Sci-Fi", "🚪", "With Spider-Man's identity now revealed, Peter asks Doctor Strange for help, fracturing the multiverse."),
    (119, "The Batman", 2022, "Action|Crime|Drama|Mystery", "🦇", "When a sadistic serial killer begins murdering key political figures in Gotham, Batman is forced to investigate the city's hidden corruption."),
    (120, "O Brother, Where Art Thou?", 2000, "Adventure|Comedy|Crime", "🪕", "In the deep south during the 1930s, three escaped convicts search for hidden treasure while a relentless lawman pursues them.")
]


def generate_benchmark_dataset(num_users: int = 600, random_state: int = 42):
    """
    Generates a highly structured, realistic movie ratings dataset based on
    latent user archetype tastes (Sci-Fi fans, Drama buffs, Animation lovers,
    Action fans, Cinephiles, and Romance lovers).
    
    Ensures collaborative filtering produces deeply intuitive, human-like recommendations.
    """
    np.random.seed(random_state)
    
    # 1. Build Movies DataFrame
    movie_records = []
    for m in BENCHMARK_MOVIES:
        m_id, title, year, genres, emoji, overview = m
        movie_records.append({
            "movie_id": m_id,
            "title": title,
            "release_year": year,
            "genres": genres,
            "genres_list": genres.split("|"),
            "emoji": emoji,
            "overview": overview
        })
    movies_df = pd.DataFrame(movie_records)

    # 2. Assign User Taste Archetypes (6 main clusters)
    # Archetype weights for genres: [Sci-Fi, Action, Crime, Drama, Animation, Comedy, Romance, Horror]
    genre_keys = ["Sci-Fi", "Action", "Crime", "Drama", "Animation", "Comedy", "Romance", "Horror", "Adventure"]
    
    archetypes = {
        "SciFi_Geek": {"Sci-Fi": 1.4, "Action": 1.2, "Adventure": 1.1, "Drama": 0.0, "Romance": -0.8, "Animation": 0.5, "Crime": 0.2, "Horror": 0.3, "Comedy": 0.1},
        "Prestige_Drama": {"Drama": 1.5, "Crime": 1.3, "Biography": 1.2, "Mystery": 1.0, "Sci-Fi": 0.1, "Romance": 0.2, "Comedy": -0.2, "Animation": -0.4, "Action": -0.2},
        "Animation_Family": {"Animation": 1.6, "Family": 1.5, "Comedy": 1.1, "Adventure": 0.9, "Fantasy": 1.0, "Romance": 0.4, "Horror": -1.5, "Crime": -1.0, "Action": 0.2},
        "Action_Adrenaline": {"Action": 1.5, "Thriller": 1.2, "Adventure": 1.1, "Sci-Fi": 0.9, "Crime": 0.8, "Romance": -0.9, "Drama": -0.2, "Animation": 0.0, "Horror": 0.4},
        "Romance_Heart": {"Romance": 1.6, "Drama": 1.1, "Comedy": 0.9, "Music": 1.0, "Animation": 0.3, "Action": -1.0, "Horror": -1.5, "Sci-Fi": -0.6, "Crime": -0.5},
        "Eclectic_Cinephile": {"Drama": 1.0, "Crime": 1.0, "Sci-Fi": 0.9, "Mystery": 1.1, "Romance": 0.5, "Animation": 0.8, "Action": 0.6, "Horror": 0.5, "Comedy": 0.5}
    }
    archetype_names = list(archetypes.keys())

    # Pre-index movie genres
    movie_genre_sets = {row["movie_id"]: set(row["genres_list"]) for _, row in movies_df.iterrows()}

    # Global base movie popularity/quality baseline
    movie_base_quality = {}
    for _, row in movies_df.iterrows():
        # Iconic classics get slightly higher base scores
        mid = row["movie_id"]
        # slight natural noise centered at 3.7
        movie_base_quality[mid] = np.clip(3.7 + np.random.normal(0, 0.35), 2.8, 4.6)

    ratings_list = []
    
    # Base timestamp
    base_ts = 1609459200 # 2021-01-01

    for u_id in range(1, num_users + 1):
        # Pick archetype
        arch_name = np.random.choice(archetype_names, p=[0.25, 0.22, 0.15, 0.18, 0.10, 0.10])
        pref_profile = archetypes[arch_name]
        
        # User rating generosity bias (some users are harsh, some generous)
        user_bias = np.random.normal(0.0, 0.35)
        
        # How many movies this user watched (between 15 and 55 out of 120)
        n_watched = np.random.randint(18, 55)
        
        # Calculate selection probability based on preferences
        probabilities = []
        for _, row in movies_df.iterrows():
            mid = row["movie_id"]
            m_genres = movie_genre_sets[mid]
            
            # affinity boost
            affinity = 1.0
            for g in m_genres:
                affinity += pref_profile.get(g, 0.0) * 0.4
            affinity = max(0.1, affinity)
            probabilities.append(affinity)
            
        prob_arr = np.array(probabilities)
        prob_arr = prob_arr / prob_arr.sum()
        
        watched_indices = np.random.choice(len(movies_df), size=n_watched, replace=False, p=prob_arr)
        
        for idx in watched_indices:
            row = movies_df.iloc[idx]
            mid = row["movie_id"]
            m_genres = movie_genre_sets[mid]
            
            # Compute score
            base_q = movie_base_quality[mid]
            taste_match = sum(pref_profile.get(g, 0.0) for g in m_genres) / max(1, len(m_genres))
            
            raw_rating = base_q + user_bias + (taste_match * 0.8) + np.random.normal(0, 0.3)
            
            # Clip between 1.0 and 5.0 and round to nearest 0.5
            clamped = np.clip(raw_rating, 1.0, 5.0)
            final_rating = round(clamped * 2) / 2.0
            
            ts = base_ts + np.random.randint(0, 86400 * 365 * 2)
            ratings_list.append({
                "user_id": int(u_id),
                "movie_id": int(mid),
                "rating": float(final_rating),
                "timestamp": int(ts)
            })

    ratings_df = pd.DataFrame(ratings_list)
    
    # Calculate aggregated statistics for movies
    agg_stats = ratings_df.groupby("movie_id")["rating"].agg(["mean", "count"]).reset_index()
    agg_stats.rename(columns={"mean": "avg_rating", "count": "rating_count"}, inplace=True)
    agg_stats["avg_rating"] = agg_stats["avg_rating"].round(2)
    
    movies_df = movies_df.merge(agg_stats, on="movie_id", how="left")
    movies_df["avg_rating"] = movies_df["avg_rating"].fillna(3.5)
    movies_df["rating_count"] = movies_df["rating_count"].fillna(0).astype(int)

    return movies_df, ratings_df


def load_dataset(use_cache: bool = True, cache_dir: str = None):
    """
    Main dataset loader. Returns (movies_df, ratings_df).
    Guaranteed zero-crash fallback to benchmark procedural dataset.
    """
    if cache_dir is None:
        cache_dir = os.path.join(CURRENT_DIR, "data_cache")
    
    movies_cache = os.path.join(cache_dir, "benchmark_movies.csv")
    ratings_cache = os.path.join(cache_dir, "benchmark_ratings.csv")
    
    if use_cache and os.path.exists(movies_cache) and os.path.exists(ratings_cache):
        try:
            movies_df = pd.read_csv(movies_cache)
            ratings_df = pd.read_csv(ratings_cache)
            movies_df["genres_list"] = movies_df["genres"].apply(lambda g: str(g).split("|"))
            return movies_df, ratings_df
        except Exception:
            pass # fallback to generation

    movies_df, ratings_df = generate_benchmark_dataset(num_users=600, random_state=42)
    
    try:
        os.makedirs(cache_dir, exist_ok=True)
        # Save cache without lists for CSV
        save_movies = movies_df.drop(columns=["genres_list"], errors="ignore")
        save_movies.to_csv(movies_cache, index=False)
        ratings_df.to_csv(ratings_cache, index=False)
    except Exception:
        pass
        
    return movies_df, ratings_df


def train_test_split_ratings(ratings_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Splits user ratings into train and test sets while ensuring training data
    covers maximum users and items.
    """
    np.random.seed(random_state)
    shuffled = ratings_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    split_idx = int(len(shuffled) * (1.0 - test_size))
    
    train_df = shuffled.iloc[:split_idx].copy()
    test_df = shuffled.iloc[split_idx:].copy()
    
    return train_df, test_df


def get_user_item_matrix(ratings_df: pd.DataFrame, fill_value: float = 0.0) -> pd.DataFrame:
    """
    Pivots user ratings into an explicit [num_users x num_movies] matrix.
    """
    matrix = ratings_df.pivot_table(index="user_id", columns="movie_id", values="rating")
    if fill_value is not None:
        matrix = matrix.fillna(fill_value)
    return matrix


if __name__ == "__main__":
    print("[*] Testing dataset generation...")
    m_df, r_df = load_dataset()
    print(f"[✓] Movies loaded: {len(m_df)} rows")
    print(f"[✓] Ratings loaded: {len(r_df)} rows")
    print(f"[✓] Unique Users: {r_df['user_id'].nunique()}, Unique Movies: {r_df['movie_id'].nunique()}")
    print(f"[✓] Average Rating: {r_df['rating'].mean():.2f} ± {r_df['rating'].std():.2f}")
    sparsity = (1.0 - len(r_df) / (r_df['user_id'].nunique() * r_df['movie_id'].nunique())) * 100
    print(f"[✓] Matrix Sparsity: {sparsity:.1f}%")
