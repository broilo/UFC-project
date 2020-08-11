# -*- coding: utf-8 -*-
import scrapy


class FightersRankSpider(scrapy.Spider):
    name = 'fighters_rank'
    allowed_domains = ['ufc.com.br']
    start_urls = ['http://ufc.com.br/']

    def parse(self, response):
        pass
