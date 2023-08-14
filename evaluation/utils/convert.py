# Copyright 2020 IBM
# Author: peter.zhong@au1.ibm.com
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the Apache 2.0 License.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# Apache 2.0 License for more details.

import distance
from apted import APTED, Config
from apted.helpers import Tree
from lxml import etree, html
from collections import deque
from tqdm import tqdm
from .parallel import parallel_process
from anytree import AnyNode, RenderTree, PreOrderIter

import re
# as per recommendation from @freylis, compile once only
CLEANER = re.compile('<.*?>') 
def cleanhtml(raw_html):
    cleantext = re.sub(CLEANER, '', raw_html)
    #print(f"raw={raw_html}, clean={cleantext}")
    return cleantext

class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)
    
    def __iter__(self):
        self.idx = -1
        return self

    def __next__(self):
        self.idx = self.idx + 1
        if self.idx == len(self.children):
            raise StopIteration
        return self.children[self.idx].decompose()

    def __len__(self):
        return len(self.children)

    def decompose(self):
        """Show tree using brackets notation"""
        if self.tag in ['td', 'th']:
            result = [ {"tag" : self.tag, "colspan" : self.colspan, "rowspan" : self.rowspan, "content" : self.content}]
        else:
            result = [ {"tag" : self.tag} ]
        for child in self.children:
            result += child.decompose()
        return result

    def bracket(self):
        """Show tree using brackets notation"""
        if self.tag in ['td', 'th']:
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % \
                     (self.tag, self.colspan, self.rowspan, self.content)
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class CustomConfig(Config):
    @staticmethod
    def maximum(*sequences):
        """Get maximum possible value
        """
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1
        """
        return float(distance.levenshtein(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        """Compares attributes of trees"""
        if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
            return 1.
        if node1.tag == 'td' or node1.tag == "th":
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.


class HTML2Token(object):
    ''' Convert HTML to NLP input tokenized string 
    '''
    def __init__(self, ignore_cells=None, structure_only=False, n_jobs=1, ignore_nodes=None):
        assert isinstance(n_jobs, int) and (n_jobs >= 1), 'n_jobs must be an integer greather than 1'
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = ignore_nodes
        self.ignore_cells = ignore_cells
        self.__tokens__ = []

    def tokenize(self, node):
        ''' Tokenizes table cells
        '''
        self.__tokens__.append('<%s>' % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != 'unk':
            self.__tokens__.append('</%s>' % node.tag)
        if node.tag not in ['td', 'th'] and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        ''' Converts HTML tree to the format required by apted
        '''
        global __tokens__
        if node.tag in ['td', 'th']:
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(node.tag,
                                 int(node.attrib.get('colspan', '1')),
                                 int(node.attrib.get('rowspan', '1')),
                                 cell, *deque())
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag not in  ['td', 'th']:
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def html_tree_to_string(self, tr):
        """
        Row 1
        COLUMN | COLUMN_VALUE [SEP] COLUMN2 | COLUMN_VALUE [SEP] ... [X_SEP]
        Row 2
        COLUMN | COLUMN_VALUE [SEP] COLUMN2 | COLUMN_VALUE [SEP] ... [X_SEP]
        """
        text = ""
        columns = []
        processing_header = False
        one_root = False
        for n_row, lst in enumerate(tr):
           
            idx = 0
            for itm in lst:
                if itm["tag"] == "thead": #or itm["tag"] == "caption":
                    processing_header = True
                    nodes = []
                    continue
                elif itm["tag"] == "tbody":
                    processing_header = False
                    
                    if len(columns):
                        i = 0
                        for x in columns:
                            j = i + x[1]
                            n = None
                            for node in nodes:
                                if node.min_index <= i <= node.max_index:
                                    n = AnyNode(id=x[0], parent=node, min_index=i, max_index=j)
                            
                            if n is None:
                                nodes.append(AnyNode(id=x[0], min_index=i, max_index=j))

                            i = j
                    one_root = len(nodes) == 1
                    #print(RenderTree(nodes[0]))
                    continue
                
                if itm["tag"]=="th" or processing_header:
                    if itm["tag"] == "tr":
                        i = 0
                        for x in columns:
                            j = i + x[1]
                            nodes.append(AnyNode(id=x[0], min_index=i, max_index=j))
                            i = j
                        columns = []
                        continue

                    if 'content' in itm:
                        colspan=int(itm['colspan'])
                        columns.append([cleanhtml(''.join(itm['content'])) if len(itm['content']) else 'Unknown', colspan])

                elif itm["tag"] == "tr":
                    if not processing_header and len(text) > 0:
                        text = text[:-6] + "[X_SEP] "
                    idx=0
                    continue
                elif 'content' in itm:
                    for f in nodes:
                        name = [node for node in PreOrderIter(f, filter_=lambda n: n.min_index <= idx < n.max_index)]
                        if len(name) > 0:
                            if one_root and len(name) > 1:
                                col_name = cleanhtml(" ".join([n.id for n in name[1:]]))
                            else:
                                col_name = cleanhtml(" ".join([n.id for n in name]))
                            
                            idx   = name[-1].max_index
                            content = cleanhtml(''.join(itm['content']))
                            #TODO: add the ignore strings as a parameter passed
                            if len(content) > 0 and content.replace(" ", "") not in self.ignore_cells:
                                text += f"{col_name} | {content} [SEP] "
                            #if len(col_name) == 0:
                            #    breakpoint()
            #if n_row > 0:
            #    text = text[:-6] + "[X_SEP] "
        #breakpoint()
        text = text.replace("'", '"') #make sure pandas doesn't break
        return text

    def convert(self, inp):
        parser = html.HTMLParser(remove_comments=True, encoding='utf-8')
        inp = html.fromstring(inp, parser=parser)
        if inp.xpath('body/table'):
            inp = inp.xpath('body/table')[0]
            if self.ignore_nodes:
                etree.strip_tags(pred, *self.ignore_nodes)
            n_nodes = len(inp.xpath(".//*"))
            tree_ = self.load_html_tree(inp)
            return self.html_tree_to_string(tree_)

    def evaluate(self, pred, true):
        ''' Computes TEDS score between the prediction and the ground truth of a
            given sample
        '''
        if (not pred) or (not true):
            return 0.0
        parser = html.HTMLParser(remove_comments=True, encoding='utf-8')
        pred = html.fromstring(pred, parser=parser)
        true = html.fromstring(true, parser=parser)
        if pred.xpath('body/table') and true.xpath('body/table'):
            pred = pred.xpath('body/table')[0]
            true = true.xpath('body/table')[0]
            if self.ignore_nodes:
                etree.strip_tags(pred, *self.ignore_nodes)
                etree.strip_tags(true, *self.ignore_nodes)
            n_nodes_pred = len(pred.xpath(".//*"))
            n_nodes_true = len(true.xpath(".//*"))
            n_nodes = max(n_nodes_pred, n_nodes_true)
            tree_pred = self.load_html_tree(pred)
            tree_true = self.load_html_tree(true)
            distance = APTED(tree_pred, tree_true, CustomConfig()).compute_edit_distance()
            return 1.0 - (float(distance) / n_nodes)
        else:
            return 0.0

    def batch_evaluate(self, pred_json, true_json):
        ''' Computes TEDS score between the prediction and the ground truth of
            a batch of samples
            @params pred_json: {'FILENAME': 'HTML CODE', ...}
            @params true_json: {'FILENAME': {'html': 'HTML CODE'}, ...}
            @output: {'FILENAME': 'TEDS SCORE', ...}
        '''
        samples = true_json.keys()
        if self.n_jobs == 1:
            scores = [self.evaluate(pred_json.get(filename, ''), true_json[filename]['html']) for filename in tqdm(samples)]
        else:
            inputs = [{'pred': pred_json.get(filename, ''), 'true': true_json[filename]['html']} for filename in samples]
            scores = parallel_process(inputs, self.evaluate, use_kwargs=True, n_jobs=self.n_jobs, front_num=1)
        scores = dict(zip(samples, scores))
        return scores


if __name__ == '__main__':
    import json
    import pprint
    with open('../evaluation/WEATHERGOV_PLUS/TablesHTML/WEATHERGOV_Table29771_Variation020.htm') as fp:
        tables = fp.read() 
    verter = HTML2Token(n_jobs=4)
    nlp_input = verter.convert(tables)
    print(nlp_input)
