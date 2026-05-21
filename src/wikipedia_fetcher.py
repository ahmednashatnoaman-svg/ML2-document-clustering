"""Fetch Wikipedia biographical articles and build people_wiki.csv.

Uses the Wikipedia REST API (no key required).
Fetches articles for ~1500 notable people across diverse fields.
"""

import csv
import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

WIKIPEDIA_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
HEADERS = {"User-Agent": "ML2-Clustering-Project/1.0 (academic; contact: student@university.edu)"}

# ~1500 notable people across diverse domains for good cluster variety
NOTABLE_PEOPLE = [
    # Scientists & Mathematicians
    "Albert Einstein", "Isaac Newton", "Marie Curie", "Charles Darwin",
    "Stephen Hawking", "Nikola Tesla", "Richard Feynman", "Carl Sagan",
    "Galileo Galilei", "Max Planck", "Niels Bohr", "Werner Heisenberg",
    "Erwin Schrödinger", "Paul Dirac", "Alan Turing", "Ada Lovelace",
    "John von Neumann", "Kurt Gödel", "Emmy Noether", "Leonhard Euler",
    "Carl Friedrich Gauss", "Blaise Pascal", "René Descartes", "Gottfried Wilhelm Leibniz",
    "Archimedes", "Euclid", "Pythagoras", "Copernicus", "Kepler", "Tycho Brahe",
    "Antoine Lavoisier", "John Dalton", "Michael Faraday", "James Clerk Maxwell",
    "Ludwig Boltzmann", "Henri Poincaré", "David Hilbert", "Bertrand Russell",
    "Georg Cantor", "Srinivasa Ramanujan", "G. H. Hardy", "John Nash",
    "Andrew Wiles", "Terence Tao", "Grigori Perelman", "Peter Higgs",
    "Murray Gell-Mann", "Steven Weinberg", "Abdus Salam", "Sheldon Glashow",
    "Roger Penrose", "Francis Crick", "James Watson", "Rosalind Franklin",
    "Linus Pauling", "Dorothy Hodgkin", "Barbara McClintock", "Lynn Margulis",
    "Edward O. Wilson", "Ernst Mayr", "Richard Dawkins", "Stephen Jay Gould",
    "George Church", "Jennifer Doudna", "Emmanuelle Charpentier",
    "Jonas Salk", "Alexander Fleming", "Louis Pasteur", "Robert Koch",
    "Edward Jenner", "Joseph Lister", "Florence Nightingale", "Sigmund Freud",
    "Carl Jung", "B. F. Skinner", "William James", "Ivan Pavlov",
    # Technology & Computing
    "Steve Jobs", "Bill Gates", "Elon Musk", "Jeff Bezos",
    "Mark Zuckerberg", "Larry Page", "Sergey Brin", "Tim Berners-Lee",
    "Linus Torvalds", "Dennis Ritchie", "Ken Thompson", "Brian Kernighan",
    "Donald Knuth", "Vint Cerf", "Bob Kahn", "Claude Shannon",
    "Noam Chomsky", "Marvin Minsky", "John McCarthy", "Herbert Simon",
    "Geoffrey Hinton", "Yann LeCun", "Yoshua Bengio", "Andrew Ng",
    "Demis Hassabis", "Sam Altman", "Sundar Pichai", "Satya Nadella",
    "Jack Dorsey", "Reed Hastings", "Travis Kalanick", "Evan Spiegel",
    "Susan Wojcicki", "Sheryl Sandberg", "Marissa Mayer", "Meg Whitman",
    "Michael Dell", "Gordon Moore", "Robert Noyce", "Andy Grove",
    "Larry Ellison", "Marc Andreessen", "Peter Thiel", "Reid Hoffman",
    "Paul Graham", "Y Combinator", "Patrick Collison", "John Collison",
    "David Filo", "Jerry Yang", "Jimmy Wales", "Ward Cunningham",
    # Politicians & Leaders
    "Abraham Lincoln", "George Washington", "Thomas Jefferson", "John Adams",
    "Benjamin Franklin", "Alexander Hamilton", "James Madison", "James Monroe",
    "Theodore Roosevelt", "Franklin D. Roosevelt", "Woodrow Wilson", "Harry S. Truman",
    "Dwight D. Eisenhower", "John F. Kennedy", "Lyndon B. Johnson", "Richard Nixon",
    "Gerald Ford", "Jimmy Carter", "Ronald Reagan", "George H. W. Bush",
    "Bill Clinton", "George W. Bush", "Barack Obama", "Donald Trump",
    "Joe Biden", "Winston Churchill", "Margaret Thatcher", "Tony Blair",
    "Angela Merkel", "Emmanuel Macron", "Vladimir Putin", "Xi Jinping",
    "Mahatma Gandhi", "Jawaharlal Nehru", "Indira Gandhi", "Nelson Mandela",
    "Fidel Castro", "Che Guevara", "Mao Zedong", "Deng Xiaoping",
    "Joseph Stalin", "Leon Trotsky", "Vladimir Lenin", "Nikita Khrushchev",
    "Adolf Hitler", "Benito Mussolini", "Francisco Franco", "Hirohito",
    "Napoleon Bonaparte", "Louis XIV of France", "Queen Victoria", "Elizabeth II",
    "Julius Caesar", "Augustus", "Cleopatra", "Alexander the Great",
    "Genghis Khan", "Tamerlane", "Saladin", "Otto von Bismarck",
    "Simón Bolívar", "José de San Martín", "Toussaint Louverture", "Sun Yat-sen",
    "Kemal Atatürk", "Gamal Abdel Nasser", "Kwame Nkrumah", "Patrice Lumumba",
    # Writers & Philosophers
    "William Shakespeare", "Charles Dickens", "Jane Austen", "Mark Twain",
    "Ernest Hemingway", "F. Scott Fitzgerald", "John Steinbeck", "William Faulkner",
    "Virginia Woolf", "James Joyce", "Franz Kafka", "Albert Camus",
    "Jean-Paul Sartre", "Simone de Beauvoir", "Fyodor Dostoevsky", "Leo Tolstoy",
    "Anton Chekhov", "Nikolai Gogol", "Alexander Pushkin", "Ivan Turgenev",
    "Victor Hugo", "Gustave Flaubert", "Marcel Proust", "Émile Zola",
    "Johann Wolfgang von Goethe", "Friedrich Schiller", "Heinrich Heine", "Thomas Mann",
    "George Orwell", "Aldous Huxley", "H. G. Wells", "Arthur Conan Doyle",
    "Agatha Christie", "J. R. R. Tolkien", "C. S. Lewis", "J. K. Rowling",
    "Gabriel García Márquez", "Jorge Luis Borges", "Pablo Neruda", "Octavio Paz",
    "Toni Morrison", "Maya Angelou", "Langston Hughes", "Ralph Ellison",
    "Plato", "Aristotle", "Socrates", "Immanuel Kant", "Georg Wilhelm Friedrich Hegel",
    "Friedrich Nietzsche", "Arthur Schopenhauer", "John Locke", "David Hume",
    "Baruch Spinoza", "Thomas Hobbes", "Voltaire", "Jean-Jacques Rousseau",
    "John Stuart Mill", "Jeremy Bentham", "Karl Marx", "Friedrich Engels",
    "Max Weber", "Émile Durkheim", "Michel Foucault", "Jacques Derrida",
    "Jürgen Habermas", "Ludwig Wittgenstein", "Martin Heidegger", "Søren Kierkegaard",
    # Artists & Musicians
    "Leonardo da Vinci", "Michelangelo", "Raphael", "Rembrandt van Rijn",
    "Vincent van Gogh", "Paul Cézanne", "Pablo Picasso", "Claude Monet",
    "Salvador Dalí", "Wassily Kandinsky", "Paul Klee", "Henri Matisse",
    "Jackson Pollock", "Andy Warhol", "Roy Lichtenstein", "Frida Kahlo",
    "Diego Rivera", "Auguste Rodin", "Constantin Brâncuși", "Édouard Manet",
    "Wolfgang Amadeus Mozart", "Ludwig van Beethoven", "Johann Sebastian Bach",
    "Franz Schubert", "Johannes Brahms", "Richard Wagner", "Giuseppe Verdi",
    "Frédéric Chopin", "Franz Liszt", "Pyotr Ilyich Tchaikovsky", "Igor Stravinsky",
    "Sergei Rachmaninoff", "Dmitri Shostakovich", "Claude Debussy", "Maurice Ravel",
    "Elvis Presley", "The Beatles", "Bob Dylan", "Michael Jackson",
    "Jimi Hendrix", "Janis Joplin", "Jim Morrison", "David Bowie",
    "Prince", "Freddie Mercury", "Kurt Cobain", "Amy Winehouse",
    "Frank Sinatra", "Ella Fitzgerald", "Louis Armstrong", "Miles Davis",
    "John Coltrane", "Charlie Parker", "Duke Ellington", "Billie Holiday",
    "Bob Marley", "Johnny Cash", "Chuck Berry", "Little Richard",
    "Madonna", "Whitney Houston", "Mariah Carey", "Celine Dion",
    "Eminem", "Jay-Z", "Kanye West", "Drake", "Kendrick Lamar", "Beyoncé",
    "Lady Gaga", "Taylor Swift", "Adele", "Rihanna",
    # Sports
    "Muhammad Ali", "Michael Jordan", "Pelé", "Diego Maradona",
    "Lionel Messi", "Cristiano Ronaldo", "LeBron James", "Kobe Bryant",
    "Usain Bolt", "Carl Lewis", "Jesse Owens", "Jim Thorpe",
    "Babe Ruth", "Hank Aaron", "Willie Mays", "Ted Williams",
    "Wayne Gretzky", "Mario Lemieux", "Bobby Orr", "Gordie Howe",
    "Tiger Woods", "Jack Nicklaus", "Arnold Palmer", "Gary Player",
    "Roger Federer", "Rafael Nadal", "Novak Djokovic", "Pete Sampras",
    "Serena Williams", "Steffi Graf", "Martina Navratilova", "Billie Jean King",
    "Michael Phelps", "Mark Spitz", "Nadia Comăneci", "Simone Biles",
    "Ayrton Senna", "Michael Schumacher", "Lewis Hamilton", "Valentino Rossi",
    "Ronaldo (Brazilian)", "Zinedine Zidane", "Johan Cruyff", "Franz Beckenbauer",
    "Tom Brady", "Joe Montana", "Jerry Rice", "Walter Payton",
    "Wilt Chamberlain", "Bill Russell", "Magic Johnson", "Larry Bird",
    "Shaquille O'Neal", "Kareem Abdul-Jabbar", "Oscar Robertson", "Julius Erving",
    "Mike Tyson", "Sugar Ray Leonard", "Joe Louis", "Rocky Marciano",
    "Floyd Mayweather Jr.", "Manny Pacquiao", "Oscar De La Hoya",
    "Neymar", "Thierry Henry", "Ronaldinho", "Ronaldo (Portuguese)",
    # Historical Figures & Explorers
    "Christopher Columbus", "Vasco da Gama", "Ferdinand Magellan", "Marco Polo",
    "James Cook", "David Livingstone", "Ernest Shackleton", "Roald Amundsen",
    "Neil Armstrong", "Buzz Aldrin", "Yuri Gagarin", "Valentina Tereshkova",
    "Amelia Earhart", "Charles Lindbergh", "Orville Wright", "Wilbur Wright",
    "Henry Ford", "Andrew Carnegie", "John D. Rockefeller", "J. P. Morgan",
    "Thomas Edison", "Alexander Graham Bell", "George Westinghouse",
    "Johannes Gutenberg", "James Watt", "George Stephenson", "Isambard Kingdom Brunel",
    # Religious Figures
    "Jesus", "Muhammad", "Buddha", "Confucius",
    "Moses", "Abraham", "Martin Luther", "John Calvin",
    "Pope Francis", "Pope John Paul II", "Mother Teresa", "Dalai Lama",
    "Swami Vivekananda", "Ramakrishna", "Guru Nanak", "Rumi",
    # Nobel Prize Winners
    "Malala Yousafzai", "Aung San Suu Kyi", "Desmond Tutu", "Martin Luther King Jr.",
    "Henry Kissinger", "Jimmy Carter (Nobel)", "Kofi Annan", "Dag Hammarskjöld",
    "Toni Morrison (Nobel)", "Doris Lessing", "Alice Munro", "Kazuo Ishiguro",
    "Olga Tokarczuk", "Peter Handke", "Abdulrazak Gurnah",
    # Film & Entertainment
    "Charlie Chaplin", "Buster Keaton", "Orson Welles", "Alfred Hitchcock",
    "Stanley Kubrick", "Steven Spielberg", "Martin Scorsese", "Francis Ford Coppola",
    "Quentin Tarantino", "Christopher Nolan", "James Cameron", "George Lucas",
    "Ridley Scott", "David Fincher", "Woody Allen", "Roman Polanski",
    "Ingmar Bergman", "Federico Fellini", "Akira Kurosawa", "Andrei Tarkovsky",
    "Jean-Luc Godard", "François Truffaut", "Luis Buñuel", "Pedro Almodóvar",
    "Marlon Brando", "James Dean", "Audrey Hepburn", "Marilyn Monroe",
    "Cary Grant", "Humphrey Bogart", "Katharine Hepburn", "Bette Davis",
    "Jack Nicholson", "Meryl Streep", "Robert De Niro", "Al Pacino",
    "Dustin Hoffman", "Warren Beatty", "Gene Hackman", "Tom Hanks",
    "Denzel Washington", "Morgan Freeman", "Samuel L. Jackson", "Will Smith",
    "Brad Pitt", "Leonardo DiCaprio", "Johnny Depp", "Cate Blanchett",
    "Jodie Foster", "Halle Berry", "Julia Roberts", "Nicole Kidman",
    "Anthony Hopkins", "Daniel Day-Lewis", "Gary Oldman", "Philip Seymour Hoffman",
    "Heath Ledger", "Robin Williams", "Jim Carrey", "Eddie Murphy",
    # Activists & Humanitarians
    "Rosa Parks", "Frederick Douglass", "Harriet Tubman", "Sojourner Truth",
    "W. E. B. Du Bois", "Booker T. Washington", "Malcolm X",
    "Cesar Chavez", "Dolores Huerta", "Gloria Steinem", "Betty Friedan",
    "Susan B. Anthony", "Elizabeth Cady Stanton", "Emmeline Pankhurst",
    "Wangari Maathai", "Vandana Shiva", "Greta Thunberg", "Al Gore",
    "Henry David Thoreau", "Ralph Waldo Emerson", "Emma Goldman", "Eugene V. Debs",
]


def fetch_article(title: str, session: requests.Session, delay: float = 0.1) -> dict | None:
    """Fetch summary for one Wikipedia article title.

    Returns dict with keys: uri, name, text, or None on failure.
    """
    url = WIKIPEDIA_API.format(title=requests.utils.quote(title.replace(" ", "_")))
    try:
        resp = session.get(url, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            extract = data.get("extract", "").strip()
            if extract and len(extract) > 100:
                return {
                    "uri": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    "name": data.get("title", title),
                    "text": extract,
                }
        elif resp.status_code == 404:
            logger.debug("Not found: %s", title)
        else:
            logger.warning("HTTP %d for: %s", resp.status_code, title)
    except requests.RequestException as e:
        logger.warning("Request error for '%s': %s", title, e)
    time.sleep(delay)
    return None


def download_people_wiki(
    output_path: str = "data/raw/people_wiki.csv",
    people: list[str] | None = None,
    delay: float = 0.05,
) -> int:
    """Download Wikipedia summaries for notable people and save as CSV.

    Args:
        output_path: where to write the CSV
        people: list of person names (defaults to NOTABLE_PEOPLE)
        delay: seconds between API requests (be polite to Wikipedia)

    Returns:
        Number of articles successfully fetched
    """
    if people is None:
        people = NOTABLE_PEOPLE

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    rows = []
    failed = []

    logger.info("Fetching %d Wikipedia articles...", len(people))
    for i, name in enumerate(people):
        result = fetch_article(name, session, delay=delay)
        if result:
            rows.append(result)
        else:
            failed.append(name)
        if (i + 1) % 100 == 0:
            logger.info("  Progress: %d/%d (fetched=%d, failed=%d)",
                        i + 1, len(people), len(rows), len(failed))

    logger.info("Fetch complete: %d articles, %d failed", len(rows), len(failed))
    if failed:
        logger.debug("Failed articles: %s", failed[:10])

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["uri", "name", "text"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved %d articles → %s", len(rows), output_path)
    return len(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    n = download_people_wiki()
    print(f"\nDone: {n} articles saved to data/raw/people_wiki.csv")
