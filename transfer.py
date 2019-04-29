import sys
import math
import types
from random import Random
from sets import Set
import Numeric

RAND = Random(1)

###########################################################################

class Tree:
    def __init__(self, label, children=()):
        self.label = label
        self.children = tuple(children)
        return

    def is_terminal(self):
        return self.children == ()

    def pprint(self, indent=''):
        print '%s%s' % (indent, self.label)
        for c in self.children:
            c.pprint(indent + '  ')
        return

    def fringe(self):
        if self.is_terminal():
            yield self.label            
        else:
            for c in self.children:
                for f in c.fringe():
                    yield f
        return

    def to_dot(self, title, writer, include_weights=False):
        writer.write('digraph %s {\n' % title)
        self._to_dot_aux(writer, 0)
        writer.write('}\n')
        writer.flush()
        return

    def _to_dot_aux(self, w, m):
        w.write('n%d [label="%s"]\n' % (m, self.label))
        n = m+1
        for i,child in enumerate(self.children):
            w.write('n%d -> n%d [label="%d"]\n' % (m, n, i+1))
            n = child._to_dot_aux(w, n)
        return n

def _tree1():
    return Tree('C', [Tree('S', [Tree('n1')]),
                      Tree('v'),
                      Tree('O', [Tree('n2')])])

def _tree2():
    return Tree('C', [Tree('S', [Tree('a1'), Tree('n1')]),
                      Tree('v'),
                      Tree('O', [Tree('a2'), Tree('n2')])])

def _test_tree():
    t1 = _tree1()
    t2 = _tree2()
    sys.stderr.write('%s\n' % list(t1.fringe()))
    sys.stderr.write('%s\n' % list(t2.fringe()))
    t1.to_dot('tree1', sys.stdout)
    t2.to_dot('tree2', sys.stdout)
    return

###########################################################################

class Forest:
    def __init__(self, identity, choices=(), label=None):
        self.identity = identity
        if label is None:
            self.label = identity
        else:
            self.label = label
        self.choices = tuple(choices)
        self.choice_weights = None
        return

    def is_terminal(self):
        return self.choices == ()

    def to_dot(self, title, writer, include_weights=False):
        writer.write('digraph %s {\n' % title)
        self._to_dot_aux(writer, {}, include_weights)
        writer.write('}\n')
        writer.flush()
        return

    def _to_dot_aux(self, w, node_map, include_weights):
        # recall memoized node
        if id(self) in node_map:
            return node_map[id(self)]
        # node is new, assign a unique integer
        n = len(node_map)
        node_map[id(self)] = n
        # format the node label
        if type(self.identity) == types.TupleType and len(self.identity) == 3:
            label = '(%s,%s,%s)' % self.identity
        else:
            label = str(self.identity)
        # write the node in dot format
        w.write('n%d [label="%s' % (n, label))
        if self.choice_weights is not None and include_weights:
            w.write('/%g' % Numeric.sum(self.choice_weights))
        w.write('"]\n')
        # deal with choice points
        if len(self.choices) == 1:
            # special case: don't draw node for single choice point
            children = self.choices[0]
            for j,child in enumerate(children):
                # recursively write the child in dot format
                cn = child._to_dot_aux(w, node_map, include_weights)
                # write an edge from the current node to the child node
                w.write('n%d -> n%d [label="%d"]\n' % (n, cn, j+1))
        else:
            for i,choice in enumerate(self.choices):
                # write the choice node in dot format
                w.write('n%dc%d [shape=box,label="%s_%d' % (n, i, label, i+1))
                if self.choice_weights is not None and include_weights:
                    w.write('/%g' % self.choice_weights[i])
                w.write('"]\n')
                # write an edge from the forest node to the choice node
                w.write('n%d -> n%dc%d [style=dashed]\n' % (n, n, i))
                # deal with children of the current choice
                for j,child in enumerate(choice):
                    # recursively write the child in dot format
                    cn = child._to_dot_aux(w, node_map, include_weights)
                    # write an edge from the choice node to the child node
                    w.write('n%dc%d -> n%d [label="%d"]\n' % (n, i, cn, j+1))
        return n

def tree2forest(tree):
    if tree.is_terminal():
        return Forest(tree.label)
    children = [ tree2forest(c) for c in tree.children ]
    return Forest(tree.label, [tuple(children)])

def forest_count_trees(node):
    return _forest_count_trees_aux(node, {})

def _forest_count_trees_aux(node, visited):
    if id(node) in visited:
        return visited[id(node)]
    if node.is_terminal():
        n = 1
    else:
        n = 0  # additive unit
        for ch in node.choices:
            m = 1  # multiplicative unit
            for child in ch:
                # children are multiplicative
                m *= _forest_count_trees_aux(child, visited)
            # choices are additive
            n += m
    visited[id(node)] = n
    return n

def _forest1():
    s = Forest('S', [(Forest('n1'),)])
    v = Forest('v')
    o = Forest('O', [(Forest('n2'),)])
    vo = Forest('VO', [(v, o)])
    ov = Forest('OV', [(o, v)])
    so = Forest('SO', [(s, o)])
    root = Forest('C', [(s,vo), (s,ov), (v,so)])
    return root

def _forest2():
    a1 = Forest('a1')
    n1 = Forest('n1')
    a2 = Forest('a2')
    n2 = Forest('n2')
    s = Forest('S', [(a1,n1), (n1,a1)])
    v = Forest('v')
    o = Forest('O', [(a2,n2), (n2,a2)])
    vo = Forest('VO', [(v, o)])
    ov = Forest('OV', [(o, v)])
    so = Forest('SO', [(s, o)])
    root = Forest('C', [(s,vo), (s,ov), (v,so)])
    return root

def _forest3():
    a1 = Forest('a1')
    n1 = Forest('n1')
    a2 = Forest('a2')
    n2 = Forest('n2')
    s = Forest('S', [(a1,n1), (n1,a1)])
    v = Forest('v')
    o = Forest('O', [(a2,n2), (n2,a2)])
    root = Forest('C', [(s,v,o), (s,o,v), (v,s,o)])
    return root

def _test_forest():
    f1 = _forest1()
    f2 = _forest2()
    f3 = _forest3()
    f1.to_dot('forest1', sys.stdout)
    f2.to_dot('forest2', sys.stdout)
    f3.to_dot('forest3', sys.stdout)
    sys.stderr.write('forest 1 contains %d trees\n' % forest_count_trees(f1))
    sys.stderr.write('forest 2 contains %d trees\n' % forest_count_trees(f2))
    sys.stderr.write('forest 3 contains %d trees\n' % forest_count_trees(f3))
    return

###########################################################################

def forest_compose_bigram(node):
    """Compose/intersect a forest with a bigram language model."""
    choices = []
    nodes = _forest_compose_bigram_aux(node, ('#',), {})
    for (a,b),L in nodes.iteritems():
        assert a == '#'
        R = Forest((b, '$', '$'))
        choices.append((L, R))
    return Forest(('#', 'root', '$'), choices)

def _forest_compose_bigram_aux(node, preceding, visited):
    if (id(node), preceding) in visited:
        return visited[id(node), preceding]
    nodes = {}
    if node.is_terminal():
        b = node.label[0]
        for a in preceding:
            nodes[a,b] = Forest((a,node.label,b))
    else:
        for choice in node.choices:
            choices = {}
            for a in preceding:
                choices[a] = { a : [[]] }
            for child in choice:
                caug = _forest_compose_bigram_aux(child,
                                                  tuple(choices.keys()),
                                                  visited)
                choices2 = {}
                for (b,c),B in caug.iteritems():
                    if c not in choices2:
                        choices2[c] = {}
                    for a,xs in choices[b].iteritems():
                        if a not in choices2[c]:
                            choices2[c][a] = []
                        choices2[c][a] += [ x+[B] for x in xs ]
                choices = choices2
            for b,d in choices.iteritems():
                for a,xs in d.iteritems():
                    if (a,b) not in nodes:
                        nodes[a,b] = []
                    nodes[a,b] += [tuple(x) for x in xs]
        for (a,b),choices in nodes.iteritems():
            nodes[a,b] = Forest((a,node.label,b), choices)
    visited[id(node),preceding] = nodes
    return nodes

def _test_compose():
    f1 = _forest1()
    sys.stderr.write('forest 1  contains %d trees\n' % forest_count_trees(f1))
    f1c = forest_compose_bigram(f1)
    f1c.to_dot('f1_bigram', sys.stdout)
    sys.stderr.write('forest 1  contains %d trees\n' % forest_count_trees(f1))
    sys.stderr.write('forest 1c contains %d trees\n' % forest_count_trees(f1c))

    f2 = _forest2()
    f2c = forest_compose_bigram(f2)
    f2c.to_dot('f2_bigram', sys.stdout)
    sys.stderr.write('forest 2  contains %d trees\n' % forest_count_trees(f2))
    sys.stderr.write('forest 2c contains %d trees\n' % forest_count_trees(f2c))
   
    f3 = _forest3()
    f3c = forest_compose_bigram(f3)
    f3c.to_dot('f3_bigram', sys.stdout)
    sys.stderr.write('forest 3  contains %d trees\n' % forest_count_trees(f3))
    sys.stderr.write('forest 3c contains %d trees\n' % forest_count_trees(f3c))
    return

###########################################################################

def faug_probability(node, params, visited={}):
    """Propagate weights bottom-up in an augmented forest and
    normalize the choice weights of each node."""
    if id(node) in visited:
        return visited[id(node)]
    t = 0
    if node.is_terminal():
        hist,w,fut = node.name
        t = params.lm(w, hist)
    else:
        choice_weights = Numeric.zeros(len(node.choices), 'd')
        for i,children in enumerate(node.choices):
            labels = [ c.label[1] for c in chldren ]
            nw = params.perm(node.label[1], labels)
            for c in children:
                nw *= faug_probability(c, params, visited)
            choice_weights[i] += nw
            t += nw
        node.choice_weights = choice_weights * (1.0 / t)
    visited[id(node)] = t
    return t

###########################################################################

def permutation_forest(tree):
    if tree.is_terminal():
        return Forest(tree.label)
    try:
        if tree.label == 'C':
            s = None
            v = None
            o = None
            for child in tree.children:
                if child.label == 'S':
                    if s is not None:
                        raise Exception()
                    s = child
                elif child.label == 'v':
                    if v is not None:
                        raise Exception()
                    v = child
                elif child.label == 'O':
                    if o is not None:
                        raise Exception()
                    o = child
                else:
                    raise Exception()
            if s is None or v is None:
                raise Exception()
            s = permutation_forest(s)
            v = permutation_forest(v)
            if o is None:
                return Forest(tree.label, [(s,v),(v,s)])
            else:
                o = permutation_forest(o)
                return Forest(tree.label, [(s,v,o),(s,o,v),(v,s,o)])
        elif tree.label == 'S' or tree.label == 'O':
            aa = []
            nn = []
            for child in tree.children:
                if child.label == 'a':
                    aa.append(child)
                elif child.label == 'n':
                    nn.append(child)
                else:
                    raise Exception()
            aa = [ permutation_forest(a) for a in aa ]
            nn = [ permutation_forest(n) for n in nn ]
            return Forest(tree.label, [tuple(aa+nn), tuple(nn+aa)])
    except:
        pass
    children = [ permutation_forest(c) for c in tree.children ]
    return Forest(tree.label, [tuple(children)])

def _test_permutation():
    t2 = _tree2()
    f = permutation_forest(t2)
    f.to_dot('permute_t2', sys.stdout)
    return

###########################################################################

if __name__ == '__main__':
    _test_tree()
    _test_forest()
    _test_compose()
    _test_permutation()

    t = _tree2()
    w = file('tree.dot', 'w')
    t.to_dot('tree', w)
    w.close()

    f1 = _forest2()
    w = file('forest1.dot', 'w')
    f1.to_dot('forest1', w)
    w.close()

    f1c = forest_compose_bigram(f1)
    w = file('forest2.dot', 'w')
    f1c.to_dot('forest2', w)
    w.close()

# eof
