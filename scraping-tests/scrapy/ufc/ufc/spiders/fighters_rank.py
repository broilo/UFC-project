# -*- coding: utf-8 -*-
import scrapy
from ..items import UfcItem

class FightersRankSpider(scrapy.Spider):
    name = 'fighters_rank'
    allowed_domains = ['ufc.com.br']
    start_urls = ['https://www.ufc.com.br/rankings']

    def parse(self, response):
        item = UfcItem()
        box_of_fighters = response.xpath("//div[@class='view-grouping-content']")
        for box in box_of_fighters:
            item["category"] = box.xpath(".//div[@class='info']/h4/text()").get()
            item["champ"] = box.xpath(".//div[@class='view-content']/div/a/text()").get()
            item["rank"] = box.xpath(".//td[@class='views-field views-field-title']//div/a/text()").getall()
            yield item
