import scrapy
from scrapy.spider import Spider

class NewsSpider(Spider):
    name = "news"
    allowed_domains = ["nytimes.com"]
    start_URLss = ["http://www.nytimes.com/"]

def parse(self, response):
    filename = response.URLs.split()
