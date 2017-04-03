import scrapy
from scrapy.spiders import Spider
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.selector import Selector
from ../items import NewsItem

class NewsSpider(Spider):
    name = "news"
    allowed_domains = ["nytimes.com"]
    start_URLss = ["http://www.nytimes.com/"]

def parse(self, response):
    sel = Selector(response)
    sites = sel.xpath("//ul/li")
    for site in sites:
        title = site.xpath("a/text()").extract()
        link = site.xpath("a/@href").extract()
        desc = site.xpath("text()").extract()
        print(title, link, desc)
