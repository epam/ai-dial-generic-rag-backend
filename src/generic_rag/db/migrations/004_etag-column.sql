alter table documents
    add column if not exists etag varchar(64) default null;
