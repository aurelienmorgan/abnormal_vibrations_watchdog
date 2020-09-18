import os

from jupyter_contrib_nbextensions.nbconvert_support import PyMarkdownPreprocessor

import nbformat, codecs
from nbconvert import HTMLExporter


#/////////////////////////////////////////////////////////////////////////////////////


class MarkdownPreprocessor(PyMarkdownPreprocessor):
    """
    :mod:`nbconvert` Preprocessor for the python-markdown nbextension.

    This :class:`~nbconvert.preprocessors.Preprocessor` replaces kernel code
    in markdown cells with the results stored in the cell metadata.
    In addition:
        - it also "re-numbers" code cells (i.e. resets
          their "execution_count" attribute).
        - it also 'ignores' the outputs of code cells
          which are "collapsed" (double-clicked).
    """

    def __init__(self) :
        self.execution_count = 0

    def preprocess_cell(self, cell, resources, index) :
        if cell.cell_type == "markdown":
            if hasattr(cell['metadata'], 'variables') :
                variables = cell['metadata']['variables']
                if len(variables) > 0:
                    cell.source = self.replace_variables(
                        cell.source, variables)
        elif cell.cell_type == "code":
            if cell['execution_count'] :
                #print(str(cell['execution_count']), file=sys.__stdout__)
                self.execution_count += 1
                cell['execution_count'] = self.execution_count
                if hasattr(cell['metadata'], 'collapsed') :
                    cell['outputs'] = []

        return cell, resources


#/////////////////////////////////////////////////////////////////////////////////////


def md_extension_to_html(notebook_full_name: str) -> None :
    """
    Convenience method to render a Jupyter Notebook into the HTML format.
    The specificity here is that such rendering shall take care of
    replacing "Markdown Extension" cells (containing {{}} fields) with
    their values.
    It also ignores collapsed code cells outputs.
    
    Parameters :
        - notebook_full_name (str) :
            The fullname of the Jupyter Notebook to be converted to HTML
            (having its Markdown cells evaluated).
    Results :
        - N.A.
    """

    full_path = (os.path.sep).join(os.path.realpath(notebook_full_name)
                                   .split(os.path.sep)[:-1])
    notebook_name = os.path.realpath(notebook_full_name).split(os.path.sep)[-1]


    html_full_name = os.path.join(full_path
                                  , '.'.join(notebook_name.split('.')[:-1]) + ".html")
    print("'" + notebook_name + "' ; '" + html_full_name + "'")

    with open( notebook_full_name ) as ipynb :
        nb = nbformat.read(ipynb, as_version=4)

    pymk = MarkdownPreprocessor()
    pymk.preprocess(nb, {})

    exporter = HTMLExporter()
    output, resources = exporter.from_notebook_node(nb)
    #print(output)
    codecs.open(html_full_name, 'w', encoding='utf-8') \
        .write(output
                   # tqdm progressbar characters - unsupported
                   .replace('â–ˆ', '█').replace('â–‹', '▋').replace('â–‰', '▋')
                   # latex - unsupported
                   .replace('$\hat{y}$', '&ycirc;')

                   # title and "code cells smaller font"
                   .replace('<title>Notebook</title>'
                           , '<title>' + '.'.join(notebook_name.split('.')[:-1]) + '</title>\n' +
                             '<style type=''text/css''>\n' +
                             'div.input_area{\n' +
                             'font-size: 8.5pt;\n' +
                             '}</style>')
              )
    print('done.')











































