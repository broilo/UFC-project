# -*- coding: utf-8 -*-
import scrapy


class ConormcgregorInfoSpider(scrapy.Spider):
    name = 'ConorMcGregor_info'
    allowed_domains = ['en.wikipedia.org']
    start_urls = ['http://en.wikipedia.org/']

    def parse(self, response):
        pass
